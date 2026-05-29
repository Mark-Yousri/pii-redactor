import React from "react";

const LABELS = {
  faces: "face",
  names: "name",
  id_numbers: "ID number",
  dates: "date",
  addresses: "address",
};

export default function SummaryBadge({ summary, verification }) {
  if (!summary) return null;

  const parts = Object.entries(LABELS)
    .filter(([key]) => summary[key] > 0)
    .map(([key, label]) => `${summary[key]} ${label}${summary[key] !== 1 ? "s" : ""}`);

  const isClean = verification?.verdict === "CLEAN";
  const isLeak = verification?.verdict === "LEAK_DETECTED";

  return (
    <div className="space-y-2">
      {parts.length > 0 ? (
        <p className="text-sm text-gray-300">
          <span className="text-green-400 font-semibold">✓</span>{" "}
          {parts.join(" · ")} masked
        </p>
      ) : (
        <p className="text-sm text-gray-500">No PII detected.</p>
      )}
      {verification && (
        <p className={`text-sm font-medium ${isClean ? "text-green-400" : isLeak ? "text-red-400" : "text-yellow-400"}`}>
          {isClean && "✓ Verified — redacted data is unrecoverable"}
          {isLeak && "⚠ Leak detected — some redacted text is still extractable"}
          {!isClean && !isLeak && "Verification status unknown"}
        </p>
      )}
    </div>
  );
}
