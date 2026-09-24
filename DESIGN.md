# Design Specification: Recent Chat Deletion

## Problem Statement
Users of "Little Miss Chatterbox" and "Little Miss Magic" can generate questions/answers and bedtime stories that persist in the backend database. Currently, there is no way to remove unwanted, sensitive, or redundant interactions. Stale or test entries remain indefinitely in the recent activity list and backend database.

## Solution
Introduce a per-item delete button (trash bin icon) on every recent chat/story card in the UI. When clicked, the user is prompted to confirm the deletion. Upon confirmation, a `DELETE` request is dispatched to the backend, removing the row from the PostgreSQL database (`interactions` or `stories` table). The frontend then re-fetches the latest entries and smoothly updates the list or shows the empty state.

## User Stories
- **US-1:** As a user of Little Miss Chatterbox, I want to delete a previous question-and-answer interaction from my recent list so that I can keep my history clean and remove obsolete prompts.
- **US-2:** As a user of Little Miss Magic, I want to delete a generated bedtime story so that I can curate the stories saved in my database.
- **US-3:** As a user, I want a confirmation step before deletion occurs so that I do not accidentally delete a saved interaction by misclicking.
- **US-4:** As an operator, I want deletion to be handled within existing serverless endpoints so that the deployment does not exceed Vercel's 12-function Hobby plan limit.

## Implementation Decisions
- **API Interfaces:**
  - `DELETE /api/history?id=<id>`: Removes record with primary key `<id>` from the `interactions` table.
  - `DELETE /api/stories?id=<id>`: Removes record with primary key `<id>` from the `stories` table.
  - Returns `200 OK` with JSON `{"status": "deleted", "id": <id>}` on success.
  - Returns `400 Bad Request` if `id` is missing, not positive, or malformed.
  - Returns `404 Not Found` if the record does not exist.
- **Service Layer Contracts:**
  - `delete_interaction(interaction_id: int) -> bool`
  - `delete_story(story_id: int) -> bool`
- **UI & Interaction:**
  - Subdued trash bin button placed in the header of each card.
  - Hover states apply Apple Liquid Glass styling tokens (glass highlight, subtle spring transition).
  - Triggers native `window.confirm` to verify intent.
  - Re-triggers `loadHistory()` / `loadStories()` on success to preserve pagination and synchronize backend state.

## Testing Decisions
- Seam 1: Service layer unit tests with mocked database pool and cursor, validating parameterized SQL execution and rowcount inspection.
- Seam 2: Handler tests verifying query string parsing, validation, and standard HTTP response codes.
- Seam 3: End-to-end browser verification checking the interactive flow, DOM synchronization, and console logs.

## Out of Scope
- Soft deletion / archiving (records are deleted permanently).
- Bulk selection / batch deletion.
- Undo snackbar after deletion.
