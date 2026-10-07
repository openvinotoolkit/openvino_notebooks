// Parent pages that are allowed to receive messages from the selector iframe.
export const TRUSTED_PARENT_ORIGINS: readonly string[] = ['https://docs.openvino.ai'];

// Returns the origin postMessage should target, or null if the embedding page
// is not trusted (in which case no message should be sent).
// document.referrer is the URL of the page that embedded this iframe.
export function resolveTargetOrigin(
  referrer: string,
  trustedOrigins: readonly string[] = TRUSTED_PARENT_ORIGINS
): string | null {
  if (!referrer) {
    return null;
  }
  try {
    const { origin } = new URL(referrer);
    return trustedOrigins.includes(origin) ? origin : null;
  } catch {
    return null;
  }
}
