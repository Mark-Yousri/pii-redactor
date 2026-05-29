import React, { useState, useCallback } from "react";
import DropZone from "./components/DropZone.jsx";
import PIIToggles from "./components/PIIToggles.jsx";
import PreviewPane from "./components/PreviewPane.jsx";
import SummaryBadge from "./components/SummaryBadge.jsx";
import { redactFile, downloadUrl } from "./api.js";

const ALL_TYPES = ["face", "name", "id_number", "date", "address"];

async function rasterizePdfClientSide(file) {
  const pdfjsLib = await import("pdfjs-dist");
  pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pages = [];
  for (let i = 1; i <= pdf.numPages; i++) {
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 1.5 });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
    pages.push(canvas.toDataURL("image/png").split(",")[1]);
  }
  return pages;
}

export default function App() {
  const [file, setFile] = useState(null);
  const [originalPages, setOriginalPages] = useState(null);
  const [enabled, setEnabled] = useState(ALL_TYPES);
  const [mode, setMode] = useState("blur");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleFile = useCallback(async (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setOriginalPages(null);
    if (f.type === "application/pdf") {
      try {
        const pages = await rasterizePdfClientSide(f);
        setOriginalPages(pages);
      } catch {
        // Non-fatal: preview just won't show
      }
    } else {
      const reader = new FileReader();
      reader.onload = (e) => setOriginalPages([e.target.result.split(",")[1]]);
      reader.readAsDataURL(f);
    }
  }, []);

  const handleRedact = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await redactFile(file, enabled, mode);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 p-6 md:p-10">
      <div className="max-w-5xl mx-auto space-y-6">
        {/* Header */}
        <div>
          <h1 className="text-2xl font-bold tracking-tight">PII Redactor</h1>
          <p className="text-sm text-gray-500 mt-1">
            True redaction — underlying text is removed, not painted over.
          </p>
        </div>

        {/* Upload */}
        <DropZone onFile={handleFile} />
        {file && (
          <p className="text-sm text-gray-400">
            Selected: <span className="text-gray-200">{file.name}</span>{" "}
            ({(file.size / 1024).toFixed(0)} KB)
          </p>
        )}

        {/* Toggles */}
        <PIIToggles
          enabled={enabled}
          onChange={setEnabled}
          mode={mode}
          onModeChange={setMode}
        />

        {/* Redact button */}
        <button
          onClick={handleRedact}
          disabled={!file || loading}
          className="rounded-lg bg-blue-600 px-6 py-2.5 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
        >
          {loading ? "Redacting…" : "Redact"}
        </button>

        {error && (
          <p className="text-sm text-red-400">Error: {error}</p>
        )}

        {/* Summary */}
        {result && (
          <SummaryBadge summary={result.summary} verification={result.verification} />
        )}

        {/* Preview */}
        <PreviewPane
          originalPages={originalPages}
          redactedPages={result?.page_previews}
          loading={loading}
        />

        {/* Download */}
        {result && (
          <a
            href={downloadUrl(result.output_file_id)}
            download="redacted.pdf"
            className="inline-block rounded-lg bg-gray-700 px-6 py-2.5 text-sm font-semibold text-white hover:bg-gray-600 transition-colors"
          >
            Download Redacted PDF
          </a>
        )}
      </div>
    </div>
  );
}
