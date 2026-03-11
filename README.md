# EduwHealth
Repository for NYUSH DIC 2025 mental health and education track

## Overview
This project is a fully local tutoring system for machine learning beginners.  
You run everything on your own machine.  
No cloud services.  
No remote APIs.

The system combines retrieval augmented generation, multi agent reasoning, and safety awareness.  
You control the models, the data, and the behavior.

## What you get
- A local tutor for machine learning fundamentals  
- Retrieval augmented generation using your own knowledge base  
- Multiple agents with clear roles  
- Risk detection powered by a local language model  
- Mental health risk assessment using pre-trained classifiers  
- Learning disabilities support with evidence-based interventions  
- Personalized intervention recommendations based on learner profile  
- Real-time adaptive tutoring matched to disability type  
- Offline execution from end to end  

## How it works
You ask a question.  
The system retrieves relevant knowledge from a vector store.  
Agents reason in parallel.  
A risk model evaluates the message.  
You receive a grounded response.

## System architecture
- Vector store built with FAISS  
- Knowledge stored as embedded text chunks  
- RAG node retrieves context before generation  
- LangGraph coordinates agent execution  
- Local language model performs reasoning and analysis  
- Risk model evaluates emotional and safety signals  

## Agents
### Tutor
- Explains concepts clearly  
- Prioritizes retrieved knowledge  

### Coach
- Supports motivation and learning persistence  
- Encourages autonomy, competence, and relatedness  

### Critic
- Reviews responses for safety  
- Flags risky patterns  

### LD Specialist
- Routes to disability-specific intervention protocols  
- Supports ADHD, executive function deficit, anxiety, and motivational disorders  
- Adapts scaffold density to the learner's severity and progress  

### Intervention Agent
- Generates a prioritised list of interventions based on risk level, disability type, and emotion signals  
- Promotes previously-successful strategies  

### Support Agent
- Provides adaptive conversational support  
- Modes: task decomposition, anxiety de-escalation, motivation repair, general support  

## Mental Health Risk Detection
Risk scoring runs on every user message.  
Feature extraction uses both a local LLM and a pre-trained mental health classifier.  
No keyword lists.  
No hand written lexicons.

The combined pipeline outputs:
- risk_level: low, moderate, high, or severe  
- risk_score from 0 to 1  
- key_indicators: anxiety, depression, positive_affect  
- intervention_priority: low, medium, high, or immediate  

## Learning Disabilities Support
The system supports the following conditions out of the box:
- ADHD  
- Executive function deficit  
- Anxiety disorder (GAD, social anxiety, panic)  
- Learned helplessness  
- Motivational disorder / academic burnout  

See `docs/learning_disabilities_support.md` for full documentation.

## Risk model
Risk scoring runs on every user message.  
Feature extraction uses a local language model.  
No keyword lists.  
No hand written lexicons.

The model outputs:
- risk_score from 0 to 1  
- risk_level as low, moderate, high, or severe  
- reasons for transparency and debugging  

## Knowledge base
You store learning material in raw text files.  
You build embeddings once.  
You reuse the vector index on every run.

Three knowledge bases are indexed:
- `kb.txt` — IELTS preparation knowledge base  
- `kb_learning_disabilities.txt` — learning disabilities research  
- `kb_interventions.txt` — evidence-based intervention strategies  

## Workflow
1. Write learning content into kb.txt or the learning disabilities KB files  
2. Run the build script to create the vector index  
3. Start the tutor application  
4. Ask questions  

## Key files
- `kb.txt`, `kb_learning_disabilities.txt`, `kb_interventions.txt`  
- `build_vector_kb.py`  
- `affect/mental_health_classifier.py`  
- `affect/ensemble_detector.py`  
- `affect/state_tracker.py`  
- `agents/intervention_agent.py`  
- `agents/support_agent.py`  
- `analytics/risk_dashboard.py`  
- `analytics/model_evaluation.py`  
- `analytics/risk_model.py`  
- `config.py`  
- `docs/learning_disabilities_support.md`  

## Local models
You choose the models.  
Ollama works well.  
CUDA acceleration supported.  
CPU fallback supported.

## Hardware support
- GPU support for embedding and inference  
- Large RAM support for fast indexing  
- NPU optional for future extensions  

## Design goals
- Full local control  
- Transparent reasoning  
- Deterministic data flow  
- Reproducible behavior  
- Research friendly structure  

## Who this is for
You study machine learning.  
You build agent systems.  
You want local execution.  
You care about safety and grounding.
You support learners with diverse needs.

## Next steps
- Add more course content  
- Expand the knowledge base  
- Train a learned risk model  
- Add evaluation scripts  
- Add a user interface  
- Expand learning disability coverage  

