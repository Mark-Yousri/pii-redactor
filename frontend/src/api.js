export async function redactFile(file, enabledTypes, redactMode) {
  const form = new FormData();
  form.append("file", file);
  form.append("enabled_types", enabledTypes.join(","));
  form.append("redact_mode", redactMode);

  const res = await fetch("/redact", { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}

export function downloadUrl(fileId) {
  return `/download/${fileId}`;
}
