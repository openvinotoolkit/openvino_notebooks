export async function copyToClipboard(text?: string): Promise<void> {
  if (!text) {
    return;
  }

  if (navigator.clipboard) {
    await navigator.clipboard.writeText(text);
  } else {
    // Test documentation page (jenkins ci deployment) uses http protocol. Navigator.clipboard API allowed in secure environment only (https://developer.mozilla.org/en-US/docs/Web/API/Clipboard).
    // To not confuse test deployment users use a workaround copy method.
    copyFallback(text);
  }
}

/**
 * Copy value to clipboard. Uses deprecated but still working document.execCommand API (https://developer.mozilla.org/en-US/docs/Web/API/Document/execCommand)
 * @param value string
 */
function copyFallback(value: string): void {
  const hiddenTextarea = createHiddenTextarea(value);
  document.body.append(hiddenTextarea);
  hiddenTextarea.select();
  document.execCommand('copy');
  hiddenTextarea.remove();
}

function createHiddenTextarea(value: string): HTMLTextAreaElement {
  const el = document.createElement('textarea');
  // Prevent zooming on iOS
  el.style.fontSize = '12pt';
  // Reset box model
  el.style.border = '0';
  el.style.padding = '0';
  el.style.margin = '0';
  // Move element out of screen horizontally
  el.style.position = 'absolute';
  el.style.left = '-9999px';
  // Move element to the same position vertically
  const yPosition = window.pageYOffset || document.documentElement.scrollTop;
  el.style.top = `${yPosition}px`;

  el.setAttribute('readonly', '');
  el.value = value;

  return el;
}
