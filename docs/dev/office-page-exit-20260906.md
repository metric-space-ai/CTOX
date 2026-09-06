# Office page-exit database lifetime

The shell owns the database shared by all mounted apps. It must not close that
database in `beforeunload`: an Office unsaved-changes guard can cancel the
navigation after that event has already run. The page then remains mounted
with permanently closed IndexedDB handles, including apps opened afterward.

Cleanup now runs on non-persisted `pagehide`, once departure is committed.
A page retained in the back-forward cache keeps its handles and health timer;
the browser suspends the page and can restore it later. App close/unsaved-work
guards are unchanged. No browser data is cleared or rewritten by this fix.

`node --test src/apps/business-os/scripts/test-shell-page-exit.mjs` exercises
the actual shell cleanup registration: cancelled departure remains writable,
cached departure remains writable, and real departure closes the database and
health timer. The cancelled-departure and actual-exit tests fail on the old
`beforeunload` listener. This is not a substitute for the live Office reload,
import/export, authentication, and security acceptance gates.

References: [beforeunload](https://developer.mozilla.org/en-US/docs/Web/API/Window/beforeunload_event)
and [pagehide/persisted](https://developer.mozilla.org/en-US/docs/Web/API/Window/pagehide_event).
