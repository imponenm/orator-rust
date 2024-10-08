# Orator-Rust
This is an endpoint for running parler-TTS concurrently on a GPU.

It's a slight modification of the parlerTTS model within Candle (https://github.com/huggingface/candle).

The main difference is the KV cache is no longer stored within the ParlerTTS struct. Instead, you create a separate cache for each thread being run.

It's worth noting this did not improve the performance of ParlerTTS, as it's a pretty computational intensive model. Still a fun project for learning though!
