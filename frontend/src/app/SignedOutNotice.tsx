import { useSyncExternalStore } from "react";

import { isSignedOut, SIGNED_OUT_MESSAGE, subscribeSignedOut } from "../api/client";

/** Shown once a request finds the hosted app's login has expired. The link is a real page
    load, because only a navigation can reach Cloudflare's login page. */
export function SignedOutNotice() {
  const signedOut = useSyncExternalStore(subscribeSignedOut, isSignedOut);
  if (!signedOut) return null;
  return (
    <div className="failbanner signedout" role="alert">
      <div className="failbanner-top">
        <i className="led err" aria-hidden />
        <b>{SIGNED_OUT_MESSAGE}</b>
        <a className="key nextstep-key" href="/">
          Sign in again
        </a>
      </div>
    </div>
  );
}
