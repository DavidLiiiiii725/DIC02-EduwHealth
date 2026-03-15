# EduwHealth 2.0 — Agent Workflow Charts

Mermaid diagrams for Reading, Writing, Speaking, and Listening agents.  
Render in [Mermaid Live Editor](https://mermaid.live/) or any Markdown viewer with Mermaid support.

---

## 1. Reading Agent Workflow

```mermaid
flowchart TB
    subgraph Input
        A[Upload IELTS PDF] --> B[Extract Paragraph Images]
        B --> C[Extract Questions from PDF]
        C --> D[Map Questions to Paragraphs]
    end

    subgraph Preflight
        D --> E[Preflight Tips<br/>AI pre-solves questions per section]
        E --> F[Create ReadingAttempt]
    end

    subgraph SectionLoop["Section-by-Section Loop"]
        F --> G[Guide Section<br/>reading_agent_guide_section]
        G --> H{User Action}
        H -->|Read & Answer| I[Submit Answer]
        H -->|Stuck| J[Request Hint<br/>reading_agent_hint]
        H -->|Ask AI Hub| K[Explain Sentence<br/>reading_agent_explain_sentence]
        
        I --> L[Evaluate Answer<br/>reading_agent_evaluate]
        L --> M{Correct?}
        M -->|Yes| N[Next Section]
        M -->|No| O[Show Strategy<br/>reading_agent_strategy]
        O --> H
        
        J --> H
        K --> H
        
        N --> P{More Sections?}
        P -->|Yes| G
        P -->|No| Q[Complete Attempt]
    end

    subgraph Output
        Q --> R[Final Strategy Summary]
    end

    style G fill:#8b9a7a,color:#fff
    style I fill:#8b9a7a,color:#fff
    style J fill:#8b7355,color:#fff
    style K fill:#8b7355,color:#fff
    style L fill:#8b9a7a,color:#fff
    style O fill:#8b7355,color:#fff
```

---

## 2. Writing Agent Workflow

```mermaid
flowchart TB
    subgraph Input
        A[User Config: topic, genre, audience,<br/>target_words, difficulty, task_size] --> B[Generate IELTS Question<br/>LLM]
        B --> C[Generate Step-by-Step Guide<br/>generate_step_by_step_guide]
    end

    subgraph TaskSetup
        C --> D[Display: Question + Steps + Checklist]
        D --> E[Parse Steps for UI<br/>writing-steps.js]
    end

    subgraph WritingLoop["Writing Loop"]
        E --> F[User: Draft in Textarea]
        F --> G{User Action}
        G -->|Get Feedback| H[api_writing_feedback]
        G -->|Mind Map| I[Open Mindmap Overlay]
        G -->|Voice Input| J[STT → Insert to Draft]
        G -->|AI Hub Quick Help| K[Chat API with draft context]
        
        H --> L[generate_adhd_writing_feedback<br/>+ mock IELTS scores]
        L --> M[Display Feedback + Score Bars]
        M --> G
        
        I --> G
        J --> G
        K --> G
    end

    subgraph Adaptive
        F -.-> N[Stuck Detection: 3min no input]
        N -.-> O[Auto-open AI Hub with<br/>continuation prompts]
        O -.-> G
    end

    style C fill:#8b9a7a,color:#fff
    style L fill:#8b9a7a,color:#fff
    style H fill:#8b7355,color:#fff
    style K fill:#8b7355,color:#fff
```

---

## 3. Speaking Agent Workflow

```mermaid
flowchart TB
    subgraph Input
        A[User Config: topic, scenario,<br/>minutes, english_level] --> B[Generate ADHD Speaking Pack<br/>generate_adhd_speaking_pack]
    end

    subgraph PackOutput
        B --> C[Markdown Pack: 30s warm-up,<br/>1min intro, 90s topic,<br/>mini dialogue, retell template,<br/>rescue phrases]
        C --> D[Split into Parts by ## headers]
        D --> E[Display Part-by-Part with<br/>Prev / Next navigation]
    end

    subgraph PracticeLoop["Practice Loop"]
        E --> F{User Action}
        F -->|Send Text to Coach| G[api_speaking_chat]
        F -->|Next Part| H[Advance to next Part]
        F -->|AI Hub Float| I[Chat API with<br/>current Part as context]
        
        G --> J[speaking_coach_reply<br/>Micro feedback + Better version + Next prompt]
        J --> K[Append to Chat History]
        K --> F
        
        H --> E
        I --> F
    end

    subgraph Output
        J -.-> L[1) Micro feedback bullets<br/>2) Better version paragraph<br/>3) Next question]
    end

    style B fill:#e8796b,color:#fff
    style J fill:#e8796b,color:#fff
    style G fill:#d96a5c,color:#fff
    style I fill:#d96a5c,color:#fff
```

---

## 4. Listening Agent Workflow

```mermaid
flowchart TB
    subgraph Input
        A[User Config: scenario,<br/>environment, goal] --> B[Generate Listening Strategy<br/>generate_adhd_listening_strategy]
    end

    subgraph StrategyOutput
        B --> C[Markdown Strategy Cards<br/>## 0) 开始前... ## 1) 任务概览...<br/>## 2) 预热 ## 3) 听的过程中...<br/>## 4) 听完之后 ## 5) 自我倡导 ## 6) 结束语]
        C --> D[Split to Cards by ## headers]
        D --> E[Display Part-by-Part with<br/>Prev / Next + Dots]
    end

    subgraph AudioSection
        B --> F[Generate Sample Passage<br/>generate_sample_listening_passage]
        F --> G[Display Passage Text]
        G --> H[TTS Playback: 0.75x / 1x / 1.25x]
    end

    subgraph LogicChain
        H --> I{User Action}
        I -->|Generate Logic Chain| J[extract_logic_chain<br/>LLM extracts structure]
        J --> K[Display Logic Chain]
        K --> I
    end

    subgraph Navigation
        E --> I
        I -->|Prev/Next Card| E
    end

    style B fill:#7a9a8e,color:#fff
    style F fill:#7a9a8e,color:#fff
    style J fill:#6a8a7e,color:#fff
```

---

## Summary Table

| Agent   | Main Entry Points                    | Key Outputs                                      |
|---------|--------------------------------------|--------------------------------------------------|
| Reading | `api_reading_upload`, `api_reading_*` | Section guidance, hints, evaluation, strategy    |
| Writing | `api_writing_generate`, `api_writing_feedback` | Step guide, IELTS question, feedback, scores |
| Speaking| `api_speaking_generate`, `api_speaking_chat`   | Practice pack (parts), coach replies             |
| Listening| `api_listening_strategy`             | Strategy cards, sample passage, logic chain     |
