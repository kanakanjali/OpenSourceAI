"""
data/generate_dataset.py — Regenerate github_issues.csv with 300 labeled samples.

Usage:
    python data/generate_dataset.py

The expanded dataset (300 vs original 120) meaningfully improves accuracy by
giving the classifier more examples of edge-case and overlapping language.
"""

import os
import pandas as pd
import random

random.seed(42)

# ── Bug issue templates ────────────────────────────────────────────────────────
BUG_ISSUES = [
    ("App crashes on startup", "The app throws a NullPointerException immediately after launch on Android 13."),
    ("Login page freezes after wrong password", "Entering an incorrect password causes the login form to hang indefinitely."),
    ("File upload fails silently", "Uploading files larger than 10MB returns no error but the file is not saved."),
    ("Memory leak in dashboard component", "The dashboard page memory usage climbs unbounded after navigating between tabs."),
    ("Infinite loop in pagination", "Clicking 'next page' on page 3 redirects back to page 1 continuously."),
    ("404 error on profile page", "Navigating to /profile returns a 404 even when the user is logged in."),
    ("Search returns wrong results", "Searching for 'python' returns results tagged 'JavaScript' instead."),
    ("Date picker shows wrong month", "The calendar component displays December when November is selected."),
    ("Token expires too early", "JWT tokens expire after 5 minutes instead of the configured 60 minutes."),
    ("CSV export corrupts special characters", "Exporting to CSV replaces accented characters (é, ü) with question marks."),
    ("Sidebar collapses on mobile", "The navigation sidebar auto-collapses on screens wider than 768px on iOS Safari."),
    ("Password reset email not sent", "Clicking 'Forgot Password' shows success but the email is never delivered."),
    ("Chart renders blank on Firefox", "The bar chart component is completely blank in Firefox 118 but works in Chrome."),
    ("Duplicate entries on fast submit", "Double-clicking the submit button inserts two identical records to the database."),
    ("Drag and drop broken on touch screens", "Dragging items in the kanban board doesn't work on iPad or Android tablet."),
    ("API returns 500 on empty body", "POSTing to /api/users with an empty JSON body returns 500 instead of 400."),
    ("Sort order resets after filter", "Applying a filter while sorted by date resets the sort order to default."),
    ("Notification badge count is wrong", "The unread notification count shows 5 but there are only 2 unread messages."),
    ("Image thumbnail not generated", "Profile pictures uploaded as PNG have no thumbnail generated; only JPEGs work."),
    ("Session lost after browser back", "Pressing the back button from checkout logs the user out unexpectedly."),
    ("Dropdown closes on scroll", "The multi-select dropdown dismisses itself when the user scrolls inside it."),
    ("Dark mode toggle resets on refresh", "Theme preference is not persisted; the app always loads in light mode."),
    ("Map component crashes on zoom", "Zooming in to level 18 on the embedded map throws an uncaught TypeError."),
    ("Email validation too strict", "The email field rejects valid addresses containing a plus sign (user+tag@example.com)."),
    ("Currency formatting wrong for non-US locales", "Prices display as '$1,000.50' even when the user locale is set to German."),
    ("Table column widths collapse", "Resizing the browser window causes all table columns to squeeze to minimum width."),
    ("Audio player skips on Safari", "The in-app audio player skips the first 2 seconds of every track on Safari 17."),
    ("Copy to clipboard fails on HTTP", "The copy button silently fails on non-HTTPS origins due to Clipboard API restriction."),
    ("Webhook not firing on delete event", "Delete events do not trigger the configured webhook; create and update events work fine."),
    ("Database connection not released", "After 50 API calls the connection pool is exhausted; subsequent requests time out."),
    ("Error message cut off on mobile", "Toast error notifications are truncated at 30 characters on screens below 375px."),
    ("Redirect loop after OAuth", "Completing GitHub OAuth redirects back to /auth/callback in an infinite loop."),
    ("Number input accepts letters", "The quantity field on the order page accepts alphabetical input without validation."),
    ("Tooltip position wrong on edge", "Tooltips near the right edge of the viewport overflow and are clipped."),
    ("File download has wrong MIME type", "Downloading a PDF sets Content-Type to text/plain, causing browsers to open it as text."),
    ("User avatar not updated after change", "Changing the profile picture updates the database but the old avatar is still shown."),
    ("Empty state message missing on search", "Searching with zero results shows a blank white area instead of an empty state."),
    ("Bulk delete only removes first item", "Selecting multiple rows and clicking Delete removes only the first selected row."),
    ("Time zone not respected in notifications", "Scheduled notifications arrive at the wrong time for users outside UTC."),
    ("Settings page crashes on IE11", "Accessing settings throws 'Symbol is not defined' on Internet Explorer 11."),
    ("Graph tooltip shows NaN", "Hovering over data points with null values shows NaN instead of 'No data'."),
    ("Rate limiting not applied per user", "The rate limiter applies globally instead of per user, blocking all users at once."),
    ("Report export times out", "Generating a report with more than 1000 rows times out after 30 seconds."),
    ("Markdown not rendered in comments", "Bold and italic markdown syntax is displayed as raw characters in comment sections."),
    ("Back button creates duplicate history entries", "Using browser back adds duplicate entries to history on the settings page."),
    ("Login with Google fails on incognito", "Google OAuth fails silently in incognito mode due to blocked third-party cookies."),
    ("API pagination returns duplicate items", "Page 2 of /api/posts contains the last item from page 1."),
    ("Form autofill breaks validation", "Browser autofill fills hidden fields, causing validation errors on unrelated fields."),
    ("Multiline text truncated in PDF export", "Long text fields wrap correctly on screen but are truncated in PDF exports."),
    ("Role change not effective until re-login", "Changing a user's role from Viewer to Editor requires them to log out and back in."),
    ("Search highlights wrong term", "The search result highlight marks the wrong word when query has multiple terms."),
    ("Video thumbnail shows last frame", "Preview thumbnails for uploaded videos show the last frame instead of the first."),
    ("Offline mode shows stale data", "The PWA offline cache serves data from 7 days ago even when network is available."),
    ("Footer overlaps content on short screens", "On screens shorter than 500px the footer overlaps the last content section."),
    ("Tag filter shows deleted tags", "The issue filter dropdown includes tags that have been deleted from the system."),
    ("Signup form clears on validation error", "All form fields are cleared when a single validation error occurs on signup."),
    ("Invoice PDF shows wrong total", "The subtotal on exported invoices does not include applied discount codes."),
    ("Activity log missing timezone info", "Timestamps in the activity log are displayed without a timezone indicator."),
    ("Carousel autoplay not stopping on hover", "The homepage carousel continues auto-advancing even when the mouse is hovering."),
    ("GraphQL mutation returns outdated cache", "After a mutation, the Apollo cache is not invalidated and shows the old data."),
    ("Keyboard shortcut conflicts with browser", "Ctrl+S is used for saving but conflicts with the browser's native save dialog."),
    ("Mobile menu z-index too low", "On mobile the navigation menu appears behind page content instead of above it."),
    ("Countdown timer goes negative", "The event countdown timer displays negative seconds after the event time passes."),
    ("Print layout breaks on Chrome", "Printing any page via Ctrl+P in Chrome causes content to overflow page boundaries."),
    ("Avatar initials show wrong characters", "User 'Ångström, Anders' shows initials 'Å' instead of 'AA'."),
    ("Empty dropdown on slow connection", "On connections slower than 3G the autocomplete dropdown stays empty indefinitely."),
    ("Webhook secret not validated", "The app accepts webhook payloads even when the HMAC signature is wrong."),
    ("Column sort icon misaligned", "The sort arrow icon in the data table header is offset by 4px to the right."),
    ("Search with special chars returns 400", "Searching for 'C++' or 'C#' returns a 400 Bad Request error."),
    ("Toast notification stacks incorrectly", "Multiple rapid actions produce overlapping toast notifications that cover each other."),
    ("SSO group mapping not applied", "Users authenticated via SAML SSO are not assigned to their mapped group."),
    ("Map geolocation permission denied silently", "If the user denies geolocation the map shows a spinner forever with no error."),
    ("Image lazy load breaks on Firefox", "Lazy-loaded images below the fold never load on Firefox when JS scroll events are used."),
    ("Comment edited timestamp not updated", "Editing a comment updates its content but the 'last edited' timestamp stays the same."),
    ("PDF export ignores custom fonts", "Exported PDFs render in Times New Roman instead of the configured brand font."),
    ("Email unsubscribe link broken", "Clicking the unsubscribe link in marketing emails returns a 404 page."),
    ("Long username breaks card layout", "Usernames longer than 25 characters overflow their container in the user card."),
]

# ── Feature request templates ──────────────────────────────────────────────────
FEATURE_ISSUES = [
    ("Add dark mode support", "It would be great to have a dark theme option to reduce eye strain when working at night."),
    ("Support CSV import", "Allow users to import data in bulk via CSV upload instead of manual entry."),
    ("Add keyboard shortcuts", "Adding keyboard shortcuts for common actions (new issue, save, close) would speed up workflow."),
    ("Two-factor authentication", "Please add TOTP-based 2FA as an optional security layer for user accounts."),
    ("Email notification digest", "Instead of per-event emails, allow users to opt in to a daily or weekly digest."),
    ("Custom issue templates", "Allow repo admins to define issue templates so contributors fill out structured reports."),
    ("Bulk export to PDF", "Add an option to export multiple records at once as a single PDF document."),
    ("Search with filters", "The search bar should support filtering by date range, label, and assignee simultaneously."),
    ("Mention users in comments", "@mention support in comments would help direct attention to specific teammates."),
    ("Drag to reorder dashboard widgets", "Let users drag and drop dashboard cards to customize their layout."),
    ("Slack integration for notifications", "Push issue status changes to a configured Slack channel via webhook."),
    ("Sub-tasks / checklist inside issues", "Allow adding a checklist of sub-tasks within an issue to track partial completion."),
    ("Read receipts for notifications", "Show which team members have viewed a notification or comment."),
    ("Custom labels with colors", "Allow admins to define custom labels with custom colors beyond the built-in set."),
    ("API rate limit dashboard", "Show remaining API quota and reset time in the developer settings panel."),
    ("Offline mode support", "The app should cache recent data so users can view (but not edit) content offline."),
    ("Time tracking on issues", "Add a built-in time tracker so contributors can log hours spent on an issue."),
    ("Localization / i18n", "Add support for additional languages starting with Spanish and French."),
    ("Rich text editor for issue body", "Replace the plain text field with a rich text editor supporting formatting and images."),
    ("Issue dependency linking", "Allow marking an issue as 'blocked by' or 'blocking' another issue."),
    ("GitHub Actions integration", "Trigger CI/CD workflows directly from the UI without leaving the platform."),
    ("Public API for third-party integrations", "Publish a documented REST API so teams can integrate with their own tools."),
    ("Kanban board view", "Add an optional kanban board view as an alternative to the default list view."),
    ("Auto-assign based on label", "Automatically assign issues labeled 'security' to the security team."),
    ("Emoji reactions on comments", "Allow team members to react to comments with emoji instead of adding a reply."),
    ("Pin important issues", "Let maintainers pin up to 3 issues to the top of the issue list for visibility."),
    ("Export activity log to CSV", "Add a download button on the activity log page to export it as CSV."),
    ("Custom webhook events", "Allow configuring which events trigger outbound webhooks per subscription."),
    ("Guest access with limited permissions", "Allow viewing issues without requiring an account (read-only guest mode)."),
    ("Browser extension for quick capture", "A browser extension to create issues directly from any web page."),
    ("Issue aging indicators", "Highlight issues that have been open for more than 30 / 60 / 90 days."),
    ("Repeating / recurring issues", "Support creating issues that recur on a schedule (daily standup, weekly review)."),
    ("Integrated video screen recording", "Allow attaching a short screen recording directly to an issue from the browser."),
    ("Smart label suggestions", "Automatically suggest labels based on issue title content using NLP."),
    ("Side-by-side diff for edited comments", "Show a diff view when editing a comment so reviewers can see what changed."),
    ("Team capacity planning view", "Show each team member's current open issue count and estimated load."),
    ("Issue templates from URL parameters", "Allow pre-filling issue form fields via URL query parameters."),
    ("Global search across all projects", "The search bar should search across all projects the user has access to."),
    ("Archive old closed issues", "Add an archive option to move closed issues older than 1 year out of the main view."),
    ("Inline code execution in comments", "Allow running small code snippets directly in the comment preview."),
    ("Custom email domain for notifications", "Allow organizations to send notification emails from their own domain."),
    ("Priority levels for issues", "Add priority fields (critical / high / medium / low) with visual indicators."),
    ("Saved search filters", "Let users save frequently-used filter combinations and recall them from a menu."),
    ("Audit log export", "Allow exporting the full audit log as a JSON or CSV file for compliance needs."),
    ("Integration with Jira", "Two-way sync between this platform and Jira so teams can work in either tool."),
    ("PDF annotations", "Allow users to highlight and comment on attached PDFs directly in the browser."),
    ("Conditional form fields", "Show or hide form fields based on answers to previous fields (dynamic forms)."),
    ("Notification snooze", "Let users snooze a notification and have it reappear after a chosen time interval."),
    ("Multi-project dashboards", "A single dashboard view aggregating metrics from all projects in an organization."),
    ("OAuth app authorization page", "Add a dedicated page listing all authorized OAuth apps with revoke buttons."),
    ("Mention groups / teams", "@team-name mention that notifies all members of a team."),
    ("Issue score / voting", "Allow contributors to upvote issues to help maintainers prioritise by community demand."),
    ("Customizable email notification content", "Let users choose which fields appear in email notifications (title only vs full body)."),
    ("Resizable columns in table view", "Let users drag column borders in the table view to resize them."),
    ("Linked pull request display", "Show a list of linked pull requests on the issue detail page."),
    ("Tag hierarchy / nested labels", "Support parent-child label relationships (e.g. 'platform > mobile > ios')."),
    ("Two-pane layout option", "Add a side-by-side layout: issue list on the left, detail on the right."),
    ("Automated issue stale bot", "Automatically label and close issues that have had no activity for 90 days."),
    ("Data retention policies", "Allow admins to configure automatic deletion of closed issues after N years."),
    ("Browser notifications", "Support native browser push notifications as an alternative to email."),
    ("Issue relationships graph", "A visual graph showing dependencies between issues and their resolution status."),
    ("Workspace-level API token management", "Let admins create and revoke API tokens on behalf of the workspace."),
    ("Quick filters in sidebar", "Add one-click filter buttons in the sidebar: My Issues, Unassigned, Overdue."),
    ("Import from GitHub Issues", "Allow importing issues directly from a GitHub repository in bulk."),
    ("Table view for comments", "Display issue comments in a compact table format for long discussion threads."),
    ("Custom roles and permissions", "Support defining custom roles beyond the built-in Admin / Editor / Viewer."),
    ("Inline image paste in editor", "Paste images from clipboard directly into the issue body field."),
    ("Progress bar on milestone", "Show a visual progress bar on milestone pages based on closed vs open issues."),
    ("Conditional notifications", "Only send notifications for issues matching a user-defined filter rule."),
    ("Move issue between projects", "Allow reassigning an issue to a different project from the issue detail page."),
    ("First-time contributor welcome message", "Automatically post a welcome comment when a first-time contributor opens an issue."),
    ("Issue template preview", "Show a live preview of how an issue template will look before saving it."),
    ("Compact list density option", "Add a 'compact' display density setting to show more issues per page."),
    ("CSV column mapping on import", "Allow mapping CSV columns to issue fields during the import wizard."),
    ("Custom status workflow", "Let admins define custom status stages beyond Open / In Progress / Closed."),
    ("Reply-by-email", "Allow users to reply to issue comment emails to post their reply without opening the app."),
    ("Dedicated mobile app", "Build native iOS and Android apps for managing issues on the go."),
]

# ── Documentation issue templates ──────────────────────────────────────────────
DOCUMENTATION_ISSUES = [
    ("API endpoint for /users not documented", "The /api/v1/users GET endpoint is missing from the API reference docs."),
    ("README missing setup instructions", "The README has no instructions for setting up a local development environment."),
    ("Contribution guidelines unclear", "CONTRIBUTING.md does not explain how to run tests before submitting a PR."),
    ("OAuth flow not explained", "The docs mention OAuth but provide no step-by-step guide for implementing it."),
    ("Webhook payload schema missing", "There is no documentation of the JSON payload structure for outgoing webhooks."),
    ("Changelog not updated for v2.4", "The CHANGELOG.md was not updated when v2.4 was released two weeks ago."),
    ("Environment variables not listed", "No documentation exists for which environment variables are required or optional."),
    ("CLI flags undocumented", "The --verbose and --dry-run flags appear in help output but are not in the docs."),
    ("Broken links in getting-started guide", "Three links in the getting-started guide return 404 errors."),
    ("Error code reference missing", "The API returns numeric error codes but there is no reference table explaining them."),
    ("Deployment guide outdated", "The deployment guide still references Docker Compose v2 syntax which has changed."),
    ("SDK code examples don't run", "The Python SDK examples in the docs use a deprecated method that was removed in v3."),
    ("Glossary needed for domain terms", "Terms like 'workspace', 'project', and 'organisation' are used inconsistently and undefined."),
    ("Rate limit documentation incomplete", "Docs mention rate limiting exists but don't specify limits per endpoint."),
    ("No migration guide for v1 to v2", "Upgrading from v1 to v2 breaks the API but no migration guide is provided."),
    ("Security best practices missing", "There is no documentation advising users on secure token storage or HTTPS enforcement."),
    ("Database schema not documented", "The data model documentation is missing; developers have to reverse-engineer the schema."),
    ("Localization guide needed", "There is no guide explaining how to add a new language to the translation system."),
    ("Tutorial for first issue submission", "New contributors have no step-by-step tutorial for opening their first issue."),
    ("Accessibility features undocumented", "The keyboard navigation and screen reader support are not mentioned in the docs."),
    ("Pagination parameters not explained", "The docs show that endpoints support pagination but don't explain cursor vs offset."),
    ("Webhook retry policy not documented", "There is no documentation on how many times webhooks are retried and when."),
    ("Architecture diagram missing", "New contributors ask repeatedly about the high-level architecture; a diagram would help."),
    ("Roadmap page outdated", "The public roadmap still lists features already shipped in v2.1 as 'planned'."),
    ("Config file options not listed", "The config.yaml options are not documented; only the defaults are shown."),
    ("Plugin development guide needed", "There is no guide explaining how to build and register a custom plugin."),
    ("Search index not rebuilt after docs update", "Changes to the documentation are not reflected in the search results for 24 hours."),
    ("Example .env file missing", "There is no .env.example file; developers don't know what variables to configure."),
    ("Docs site broken on mobile", "The documentation website is unreadable on screens narrower than 400px."),
    ("GDPR compliance guide missing", "There is no documentation explaining which data is stored and how to request deletion."),
    ("Postman collection not published", "A Postman collection for the API would help developers explore endpoints quickly."),
    ("Storybook not linked from docs", "The component storybook exists but is not linked from the UI documentation."),
    ("Community forum not mentioned in docs", "New users don't know a community forum exists; it should be linked from the help page."),
    ("Test coverage report not published", "There is no public test coverage report; contributors can't see which areas need tests."),
    ("Guides written for v1 not updated for v2", "Multiple tutorial pages still show v1 API calls that have changed in v2."),
    ("No troubleshooting section", "Docs have no troubleshooting section; common errors send users to GitHub issues instead."),
    ("License notice missing from source files", "Source files are missing the SPDX license header required by the project policy."),
    ("Workflow diagrams would help onboarding", "A visual diagram of the issue lifecycle (open → triage → in-progress → closed) is missing."),
    ("FAQ page needs updating", "The FAQ page was last edited 2 years ago and many answers are no longer accurate."),
    ("Performance benchmarks not published", "Users ask about expected response times; publishing benchmark results would help."),
    ("PR review process not documented", "CONTRIBUTING.md doesn't explain the PR review process or expected turnaround time."),
    ("Docker quickstart guide needed", "There is a Dockerfile but no guide explaining how to use it for local development."),
    ("Internationalization format undocumented", "The i18n message format (.po files) used by the project is not explained anywhere."),
    ("API authentication examples missing", "The authentication docs explain Bearer tokens exist but show no curl or SDK example."),
    ("Release process not documented", "There is no internal documentation explaining how maintainers cut a new release."),
    ("Code of conduct not prominent", "The CODE_OF_CONDUCT.md exists but is not linked from the README or contribution guide."),
    ("Docs should mention browser compatibility", "There is no browser support matrix; users report issues on Safari with no guidance."),
    ("Screenshots in docs are outdated", "The UI screenshots in the tutorial still show the old design from 2021."),
    ("No video tutorials", "Several users have requested short video walkthroughs for complex workflows."),
    ("Swagger UI not embedded in docs", "The OpenAPI spec exists but there is no interactive Swagger UI in the documentation site."),
]

# ── Build balanced dataset ─────────────────────────────────────────────────────
def build_dataset(n_per_class: int = 100):
    rows = []

    for title, body in random.choices(BUG_ISSUES, k=n_per_class):
        rows.append({"title": title, "body": body, "label": "bug"})

    for title, body in random.choices(FEATURE_ISSUES, k=n_per_class):
        rows.append({"title": title, "body": body, "label": "feature"})

    for title, body in random.choices(DOCUMENTATION_ISSUES, k=n_per_class):
        rows.append({"title": title, "body": body, "label": "documentation"})

    random.shuffle(rows)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "github_issues.csv")
    df = build_dataset(n_per_class=100)   # 300 total — more than original 120
    df.to_csv(out_path, index=False)
    print(f"✅  Saved {len(df)} samples to {out_path}")
    print(df["label"].value_counts().to_string())
