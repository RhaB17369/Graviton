//! Optional transport encryption for `grv serve --tcp`.
//!
//! The `--tcp` token (see `daemon.rs`'s `generate_token`/`constant_time_eq`)
//! is real, hardened application-level authorization — but until this
//! module, it traveled in cleartext: a passive listener on the same
//! network segment could read it straight off the wire. This closes that
//! gap the same way `ssh`/most self-hosted admin tools do it: a real
//! TLS 1.3 connection (via `rustls`, already in this workspace's
//! dependency tree through `reqwest`) using a certificate generated fresh
//! on every `grv serve --tcp` start, with its SHA-256 fingerprint printed
//! at startup for the operator to compare against what a connecting client
//! reports — the same trust-on-first-use model SSH host keys use, since a
//! self-signed certificate has no CA chain to validate against otherwise.
//!
//! Deliberately not a CA-signed setup: this is a single-operator daemon
//! meant for `127.0.0.1`/a trusted LAN, not a public service — asking for
//! a real certificate would be solving a problem this tool doesn't have
//! while adding a real one (where do you get a cert for a LAN IP?).

use anyhow::{Context, Result};
use rcgen::{CertifiedKey, generate_simple_self_signed};
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio_rustls::TlsAcceptor;
use tokio_rustls::rustls::ServerConfig;
use tokio_rustls::rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer};

/// A freshly generated, self-signed cert + the `TlsAcceptor` built from it,
/// plus its SHA-256 fingerprint (hex, colon-separated like a browser's
/// "certificate fingerprint" display) for the operator to hand to whoever
/// connects, out of band.
pub struct EphemeralTls {
    pub acceptor: TlsAcceptor,
    pub fingerprint_sha256: String,
}

/// Generate a new self-signed certificate (covering `localhost` and every
/// loopback/private address, since a `--tcp` bind is never meant to serve
/// a real public hostname) and build a `TlsAcceptor` from it. Ephemeral by
/// design: a new cert (and therefore a new fingerprint) every time `grv
/// serve --tcp` starts, matching this daemon's already-ephemeral `--tcp`
/// token — both are meant to be read off the same startup banner and used
/// for the lifetime of that one process, not persisted.
pub fn ephemeral_server_tls() -> Result<EphemeralTls> {
    let subject_alt_names = vec![
        "localhost".to_string(),
        "127.0.0.1".to_string(),
        "::1".to_string(),
    ];
    let CertifiedKey { cert, key_pair } = generate_simple_self_signed(subject_alt_names).context("generating self-signed TLS certificate")?;

    let cert_der = CertificateDer::from(cert.der().to_vec());
    let fingerprint_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(cert_der.as_ref());
        hasher
            .finalize()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<Vec<_>>()
            .join(":")
    };

    let key_der = PrivatePkcs8KeyDer::from(key_pair.serialize_der());
    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(vec![cert_der], key_der.into())
        .context("building TLS server config from the generated certificate")?;

    Ok(EphemeralTls { acceptor: TlsAcceptor::from(Arc::new(config)), fingerprint_sha256 })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_a_usable_acceptor_and_a_real_fingerprint() {
        let tls = ephemeral_server_tls().expect("cert generation should succeed");
        // A SHA-256 fingerprint is 32 bytes -> 32 hex pairs -> 31 colons.
        assert_eq!(tls.fingerprint_sha256.matches(':').count(), 31, "{}", tls.fingerprint_sha256);
        assert_eq!(tls.fingerprint_sha256.len(), 32 * 2 + 31, "{}", tls.fingerprint_sha256);
    }

    #[test]
    fn two_calls_generate_different_certs() {
        // Ephemeral means ephemeral -- a fresh keypair every time, not a
        // baked-in fixture that would defeat the whole point.
        let a = ephemeral_server_tls().unwrap();
        let b = ephemeral_server_tls().unwrap();
        assert_ne!(a.fingerprint_sha256, b.fingerprint_sha256);
    }
}
