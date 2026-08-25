//! Real-time web access for `grv run`: `web_search` and `web_fetch`.
//!
//! No API key, by deliberate choice (see the conversation this shipped
//! from): a keyed search API (Brave/Tavily) gives better results but
//! requires the user to go get and manage a key before the tool is usable
//! at all — this works out of the box. The cost of that choice is
//! fragility: we're parsing DuckDuckGo's actual result HTML with hand-rolled
//! string scanning (no HTML parser crate, same philosophy as `graviton_core`'s
//! hand-rolled TOML), so a markup redesign on their end breaks this until
//! updated. Mojeek and Startpage were tried and rejected — Mojeek serves a
//! captcha to scripted requests, Startpage refused the connection outright;
//! DuckDuckGo's `lite` and `html` endpoints (two different endpoints, same
//! engine — not real backend diversity, but resilient to one being
//! throttled) both returned clean, parseable results in testing.

use anyhow::{bail, Context, Result};

const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) GRAVITON/0.6";
const MAX_RESULTS: usize = 8;

fn client() -> reqwest::Client {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .timeout(std::time::Duration::from_secs(20))
        .build()
        .unwrap_or_default()
}

/// Fetch a URL and return its visible text (scripts/styles stripped, tags
/// stripped, whitespace collapsed) — for reading a specific page (a doc, an
/// advisory, a result from `search`).
pub async fn fetch(url: &str) -> Result<String> {
    let resp = client().get(url).send().await.with_context(|| format!("fetching {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        bail!("HTTP {status} fetching {url}");
    }
    let body = resp.text().await.context("reading response body")?;
    Ok(html_to_text(&body))
}

/// Search the web via DuckDuckGo (lite endpoint first, html endpoint as a
/// fallback if that one is empty/unreachable) and return up to
/// `MAX_RESULTS` "title\n  url" entries with snippets.
pub async fn search(query: &str) -> Result<String> {
    let encoded = percent_encode(query);
    let lite_url = format!("https://lite.duckduckgo.com/lite/?q={encoded}");
    if let Ok(body) = get_body(&lite_url).await {
        let results = parse_duckduckgo_results(&body);
        if !results.trim().is_empty() {
            return Ok(results);
        }
    }
    let html_url = format!("https://html.duckduckgo.com/html/?q={encoded}");
    let body = get_body(&html_url).await.context("both DuckDuckGo endpoints failed")?;
    let results = parse_duckduckgo_results(&body);
    if results.trim().is_empty() {
        bail!("no results parsed — DuckDuckGo's result markup may have changed");
    }
    Ok(results)
}

async fn get_body(url: &str) -> Result<String> {
    let resp = client().get(url).send().await.with_context(|| format!("querying {url}"))?;
    let status = resp.status();
    if !status.is_success() {
        bail!("HTTP {status} from {url}");
    }
    resp.text().await.context("reading search response body")
}

/// DuckDuckGo's result anchors (both `lite` and `html` endpoints) look like
/// `<a ... href="//duckduckgo.com/l/?uddg=<percent-encoded-real-url>&rut=...">Title</a>`
/// — walk the string for `uddg=` markers rather than pulling in an HTML
/// parser crate for two known, stable endpoints.
fn parse_duckduckgo_results(html: &str) -> String {
    let mut out = String::new();
    let mut count = 0;
    let mut cursor = html;
    while let Some(rel) = cursor.find("uddg=") {
        cursor = &cursor[rel + "uddg=".len()..];
        let Some(amp) = cursor.find('&') else { break };
        let encoded_url = cursor[..amp].to_string();
        let Some(gt) = cursor.find('>') else { break };
        let after_gt = &cursor[gt + 1..];
        let Some(close) = after_gt.find("</a>") else { break };
        let title_raw = &after_gt[..close];
        cursor = &after_gt[close + "</a>".len()..];

        if title_raw.len() > 400 {
            continue; // not a real result title, skip without losing our place
        }
        let title = html_unescape(&strip_tags(title_raw));
        if title.trim().is_empty() {
            continue;
        }
        let url = percent_decode(&encoded_url);
        count += 1;
        out.push_str(&format!("{count}. {}\n   {}\n", title.trim(), url));
        if count >= MAX_RESULTS {
            break;
        }
    }
    out
}

fn html_to_text(html: &str) -> String {
    let no_scripts = strip_block(html, "<script", "</script>");
    let no_styles = strip_block(&no_scripts, "<style", "</style>");
    let text = html_unescape(&strip_tags(&no_styles));
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_block(s: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(start) = rest.find(open) {
        out.push_str(&rest[..start]);
        match rest[start..].find(close) {
            Some(end) => rest = &rest[start + end + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

fn strip_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn html_unescape(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#x27;", "'")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(b) = u8::from_str_radix(std::str::from_utf8(&bytes[i + 1..i + 3]).unwrap_or(""), 16) {
                out.push(b);
                i += 3;
                continue;
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.as_bytes() {
        match *b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(*b as char),
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A trimmed real fragment of lite.duckduckgo.com/lite/'s result markup
    /// (captured while building this), so the parser is tested against DDG's
    /// actual shape without a live network call.
    const SAMPLE_RESULT_HTML: &str = r#"
        <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.rs%2Ftokio%2Flatest%2Ftokio%2Fsync%2Fstruct.Semaphore.html&amp;rut=abc123" class='result-link'>Semaphore in tokio::sync - Rust - Docs.rs</a>
        <a rel="nofollow" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fgithub.com%2Ftokio-rs%2Ftokio&amp;rut=def456" class='result-link'>tokio-rs/tokio - GitHub</a>
    "#;

    #[test]
    fn parses_duckduckgo_result_markup() {
        let out = parse_duckduckgo_results(SAMPLE_RESULT_HTML);
        assert!(out.contains("Semaphore in tokio::sync - Rust - Docs.rs"), "got:\n{out}");
        assert!(out.contains("https://docs.rs/tokio/latest/tokio/sync/struct.Semaphore.html"), "got:\n{out}");
        assert!(out.contains("tokio-rs/tokio - GitHub"), "got:\n{out}");
        assert!(out.contains("https://github.com/tokio-rs/tokio"), "got:\n{out}");
    }

    #[tokio::test]
    #[ignore] // network — run explicitly with `cargo test -- --ignored`
    async fn live_search_and_fetch() {
        let results = search("rust tokio semaphore").await.expect("search failed");
        assert!(!results.trim().is_empty());
        println!("search results:\n{results}");

        let text = fetch("https://example.com").await.expect("fetch failed");
        assert!(text.contains("Example Domain"), "got: {text}");
        println!("fetched text:\n{text}");
    }
}
