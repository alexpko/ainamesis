# aInamnesis Demo

## Overview
This document outlines the demonstration of **aInamnesis**, an offline-first, intelligent system designed to transform clinical consultations into structured medical documentation.

## Demo Walkthrough (3-Minute Script)

### [0:00–0:45] The Problem
*   **Narrator**: "As a GP in an intercultural clinical setting, I face a constant struggle: balancing deep, patient-centered communication with the grueling administrative requirement of clinical documentation. Standard AI tools are often cloud-based, which is a non-starter for sensitive, private medical data."

### [0:45–1:45] The Solution
*   **Narrator**: "aInamnesis solves this by moving the entire processing pipeline locally. Here, I’m using a Whisper-based transcription engine to capture the consultation. As the dialogue unfolds, the system doesn't just record text—it parses it using local LLM nodes, extracting clinical metrics directly into structured SOAP notes."

### [1:45–2:30] Architecture & IBM Bob Integration
*   **Narrator**: "I utilized IBM Bob as my primary development tool to orchestrate this pipeline. IBM Bob was instrumental in mapping the logic between our local transcription stream and the structured data output. It helped me iterate quickly on the prompting strategy, ensuring that key clinical markers like the ICE algorithm (Ideas, Concerns, Expectations) are extracted accurately and saved directly to my local markdown notes."

### [2:30–3:00] Impact
*   **Narrator**: "The result is a system that automates the documentation heavy-lifting, allowing me to focus on the patient, not the screen—all while keeping sensitive data 100% offline and secure."

---

## Technical Setup for Demo
1.  **Transcription**: Local Whisper instance running on dedicated hardware.
2.  **LLM Processing**: Local Ollama instance (e.g., Llama 3 or Mistral).
3.  **Integration**: IBM Bob-orchestrated Python scripts for data mapping.
4.  **Storage**: Local Markdown-based integration (e.g., Obsidian) for immediate practitioner review.

*Note: The actual demo video link should be inserted into your BeMyApp project submission page as requested.*
