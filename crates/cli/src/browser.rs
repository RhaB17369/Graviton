//! Thin wrapper around chromiumoxide (CDP) for the `browser_*` agent tools.
//! Launches the system Chromium headlessly on first use and keeps one page
//! alive for the duration of a `grv run` session.

use anyhow::{Context, Result};
use chromiumoxide::browser::{Browser, BrowserConfig};
use chromiumoxide::page::Page;
use futures_util::StreamExt;
use std::sync::{Arc, Mutex};

/// Common install paths for Chromium/Chrome on Linux; chromiumoxide's own
/// auto-detection doesn't cover every distro layout (Arch/Garuda installs
/// to /usr/bin/chromium).
const CANDIDATE_EXECUTABLES: &[&str] = &[
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
];

pub struct BrowserSession {
    _browser: Browser,
    page: Page,
    console_log: Arc<Mutex<Vec<String>>>,
    _handler_task: tokio::task::JoinHandle<()>,
}

impl BrowserSession {
    pub async fn launch() -> Result<Self> {
        let found = CANDIDATE_EXECUTABLES
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .copied();
        let mut builder = BrowserConfig::builder();
        if let Some(exe) = found {
            builder = builder.chrome_executable(exe);
        }
        let config = builder
            .no_sandbox()
            .build()
            .map_err(|e| anyhow::anyhow!("building browser config: {e}"))?;

        let (browser, mut handler) = Browser::launch(config)
            .await
            .context("launching headless Chromium — is chromium/google-chrome installed?")?;

        let handler_task = tokio::spawn(async move {
            while let Some(_event) = handler.next().await {
                // Draining the handler stream is required for the browser
                // connection to make progress; we don't need the raw CDP
                // events here (console capture uses its own listener below).
            }
        });

        let page = browser.new_page("about:blank").await.context("opening initial page")?;

        use chromiumoxide::cdp::js_protocol::runtime::{EnableParams, EventConsoleApiCalled};
        let _ = page.execute(EnableParams::default()).await; // enable Runtime domain events

        let console_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        if let Ok(mut events) = page.event_listener::<EventConsoleApiCalled>().await {
            let log = console_log.clone();
            tokio::spawn(async move {
                while let Some(ev) = events.next().await {
                    let text = ev
                        .args
                        .iter()
                        .filter_map(|a| {
                            a.value
                                .as_ref()
                                .map(|v| v.to_string())
                                .or_else(|| a.description.clone())
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    log.lock().unwrap().push(format!("[{:?}] {}", ev.r#type, text));
                }
            });
        }

        Ok(Self { _browser: browser, page, console_log, _handler_task: handler_task })
    }

    pub async fn navigate(&self, url: &str) -> Result<String> {
        self.page.goto(url).await.with_context(|| format!("navigating to {url}"))?;
        let _ = self.page.wait_for_navigation().await;
        let title = self.page.evaluate("document.title").await.ok()
            .and_then(|r| r.into_value::<String>().ok())
            .unwrap_or_default();
        Ok(format!("navigated to {url}\ntitle: {title}"))
    }

    pub async fn eval(&self, script: &str) -> Result<String> {
        let result = self.page.evaluate(script).await.context("evaluating script")?;
        match result.value() {
            Some(v) => Ok(serde_json::to_string_pretty(v).unwrap_or_else(|_| v.to_string())),
            None => Ok("(no return value)".to_string()),
        }
    }

    pub async fn screenshot(&self, out_path: &std::path::Path) -> Result<String> {
        use chromiumoxide::page::ScreenshotParams;
        let bytes = self
            .page
            .screenshot(ScreenshotParams::builder().full_page(true).build())
            .await
            .context("capturing screenshot")?;
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(out_path, &bytes)
            .with_context(|| format!("writing screenshot to {}", out_path.display()))?;
        Ok(format!("saved {} ({} bytes)", out_path.display(), bytes.len()))
    }

    pub fn console_logs(&self) -> String {
        let log = self.console_log.lock().unwrap();
        if log.is_empty() {
            "(no console output captured)".to_string()
        } else {
            log.join("\n")
        }
    }
}
