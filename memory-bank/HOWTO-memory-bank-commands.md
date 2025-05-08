# How To Use Memory Bank Commands with Cline

This document outlines the key commands and workflows for interacting with Cline using the Memory Bank system. The Memory Bank helps Cline maintain project context across sessions.

## Core Commands

1.  **`initialize memory bank`**
    *   **When to use:** When starting a *new* project with Cline where Memory Bank files do not yet exist.
    *   **What it does:** Instructs Cline to create the initial set of core Memory Bank files (`projectbrief.md`, `productContext.md`, etc.) in a `memory-bank/` directory, populating them based on its initial understanding of the project goal.

2.  **`follow your custom instructions`** (or implicitly at the start of a task)
    *   **When to use:** At the beginning of *every* new task or conversation after a break or context reset.
    *   **What it does:** Instructs Cline to read *all* existing files within the `memory-bank/` directory to rebuild its understanding and context for the project before starting the task. (This is governed by the instructions in `.clinerules`).

3.  **`update memory bank`**
    *   **When to use:** During a task, when you want to ensure Cline captures the current state, recent decisions, or significant changes before proceeding, especially if the conversation has been long or complex, or before ending a session.
    *   **What it does:** Triggers Cline to perform a full review of *all* Memory Bank files, update them with the latest context, learnings, progress, and next steps based on the ongoing conversation, and confirm the updates. Focus is often on `activeContext.md` and `progress.md`.

## Core Workflows

### Starting a New Project

1.  Create a `memory-bank/` folder (optional, Cline can create it).
2.  Have a basic project goal in mind.
3.  Tell Cline: **`initialize memory bank`**.
4.  Review and refine the files Cline creates.

### Starting a New Task / Continuing Work

1.  Start a new conversation/task with Cline.
2.  Tell Cline: **`follow your custom instructions`** (or rely on `.clinerules` to trigger this automatically).
3.  Cline will confirm it has read the Memory Bank files.
4.  Proceed with your task request.

### During a Task

-   Work with Cline in **Plan Mode** for strategy and **Act Mode** for execution.
-   If the conversation becomes long or context seems unclear, ask Cline to **`update memory bank`**.
-   Cline should also proactively update files after significant changes or discoveries.

### Managing Context Limits

-   If Cline's responses degrade or it seems to lose track of earlier details, the context window might be full.
-   Ask Cline to **`update memory bank`** to save the current state.
-   Start a new task/conversation.
-   Begin the new conversation by asking Cline to **`follow your custom instructions`**.

**Remember:** Consistent use of these commands and workflows ensures Cline retains project knowledge effectively across sessions.
