# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Setup and Running
```bash
# Install dependencies
uv sync

# Run the application (preferred)
./run.sh

# Manual start
cd backend && uv run uvicorn app:app --reload --port 8000
```

### Environment Configuration
- Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`
- The app runs on http://localhost:8000 with API docs at http://localhost:8000/docs

## Architecture Overview

This is a **Retrieval-Augmented Generation (RAG) system** for course materials with a tool-based AI architecture.

### Core Processing Flow
1. **Document Processing Pipeline**: `DocumentProcessor` parses course files with structured format (Course Title/Link/Instructor, then Lesson sections), chunks text with sentence-based overlapping, and creates `CourseChunk` objects with contextual metadata
2. **Vector Storage**: `VectorStore` uses ChromaDB + sentence-transformers for semantic search with course/lesson filtering
3. **Tool-Based AI**: Claude decides when to search vs use general knowledge via `CourseSearchTool` through `ToolManager`
4. **Session Management**: `SessionManager` maintains conversation context across queries

### Key Components

**Backend (`/backend/`)**:
- `rag_system.py` - Main orchestrator coordinating all components
- `ai_generator.py` - Anthropic Claude integration with tool calling
- `search_tools.py` - Tool definitions and execution (CourseSearchTool)
- `document_processor.py` - Course document parsing and text chunking
- `vector_store.py` - ChromaDB vector operations with SearchResults wrapper
- `session_manager.py` - Conversation history management
- `config.py` - Configuration dataclass with environment variables

**Frontend (`/frontend/`)**:
- Vanilla HTML/CSS/JS with markdown parsing for responses
- Handles session state, loading states, and source attribution UI

### Document Format Expected
```
Course Title: [title]
Course Link: [url]
Course Instructor: [instructor]

Lesson 0: Introduction
Lesson Link: [lesson_url]
[content...]

Lesson 1: Next Topic
[content...]
```

### Configuration
Key settings in `config.py`:
- `CHUNK_SIZE: 800` - Text chunk size for vector storage
- `CHUNK_OVERLAP: 100` - Character overlap between chunks
- `MAX_RESULTS: 5` - Search results limit
- `EMBEDDING_MODEL: "all-MiniLM-L6-v2"` - Sentence transformer model

### AI Tool System
- `CourseSearchTool` provides semantic search with optional course_name and lesson_number filtering
- Claude automatically decides when to use tools vs answer from general knowledge
- Sources are tracked and displayed in the frontend UI
- Tool results are formatted with course/lesson context headers

### Data Models
- `Course` - Course metadata with lessons list
- `Lesson` - Lesson number, title, optional link
- `CourseChunk` - Text chunk with course/lesson references and chunk index
- `SearchResults` - Container for ChromaDB results with error handling

### Session Flow
RAGSystem.query() → retrieve history → AIGenerator with tools → potential tool execution via ToolManager → response with sources → update session history

### Database
- ChromaDB persistent storage in `./chroma_db/`
- Automatic document loading from `../docs/` on startup
- Course deduplication based on course titles
- always use uv to run the server, do not use pip directly