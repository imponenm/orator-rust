# Orator-Rust
This is an endpoint for running a text-to-speech endpoint. It's written in Rust and uses the Axum webserver framework to get around issues related to Python/FastAPI/GIL, which make it impossible to stream bytes back to the client as the model generates them, due to the blocking nature of FastAPI.

