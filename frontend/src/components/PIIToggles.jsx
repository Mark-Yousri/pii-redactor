import React from "react";

const PII_LABELS = {
  face: "Faces",
  name: "Names",
  id_number: "ID Numbers",
  date: "Dates",
  address: "Addresses",
};

export default function PIIToggles({ enabled, onChange, mode, onModeChange }) {
  const toggle = (type) => {
    if (enabled.includes(type)) onChange(enabled.filter((t) => t !== type));
    else onChange([...enabled, type]);
  };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-2">
        {Object.entries(PII_LABELS).map(([type, label]) => (
          <button
            key={type}
            onClick={() => toggle(type)}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors
              ${enabled.includes(type)
                ? "bg-blue-600 text-white"
                : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-3 text-sm">
        <span className="text-gray-400">Redact style:</span>
        {["blur", "box"].map((m) => (
          <button
            key={m}
            onClick={() => onModeChange(m)}
            className={`rounded px-3 py-1 capitalize transition-colors
              ${mode === m ? "bg-gray-600 text-white" : "bg-gray-800 text-gray-400 hover:bg-gray-700"}`}
          >
            {m}
          </button>
        ))}
      </div>
    </div>
  );
}
