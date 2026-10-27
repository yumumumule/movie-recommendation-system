# GroupLens HTTPS certificate chain

`grouplens.pem` contains public CA certificates only. The downloader adds this
chain to a request-local SSL context to support the official GroupLens server's
2026 certificate chain on machines with older trust stores. Certificate validity,
signatures, and the `files.grouplens.org` hostname are verified normally.
No system trust settings are changed.

Sources:

- Root: [emSign's HTTPS repository](https://repository.emsign.com/certs/EMIncommonOVG3C.p7c),
  retrieved over verified HTTPS and converted from PKCS#7 to PEM.
- Intermediate: the server certificate's AIA URL,
  `http://repository.emsign.com/certs/EEEMIncommonOVG3C.crt`.
  Its signature is verified against the root; HTTP transport confers no trust.

Root certificate: **emSign Root TLS CA - G3**.
DER SHA-256: `7dd78d5f4f13459a83dff9abbba62edbaf6f2d102bf257fd712f4d9f2746ed8d`.
Intermediate: **InCommon Intermediate CA - OVG3C**.
Public certificate material is supplied by the certificate authorities and is
not part of the independently implemented recommendation algorithms.
