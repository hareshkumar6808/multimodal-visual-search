import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { AnalyzeResponse } from "../types/api";
import { TracePanel } from "./TracePanel";

const response: AnalyzeResponse = {
  request_id: "abc-123",
  answer: "Returned response",
  suggested_actions: [],
  mir_summary: { primary_modality: "code", confidence: 0.94 },
  route: { intent: "debug", expert: "code-expert", provider: "provider-name", reason_code: "CODE_DEBUG_TEXT_SUFFICIENT" },
  trace: [
    { stage: "capture_received", status: "complete" },
    { stage: "perception", status: "complete", message: "Code detected" },
    { stage: "routing", status: "complete", message: "Code Expert selected" },
  ],
  metrics: { latency_ms: 1200, perception_ms: 250, routing_ms: 2, provider_ms: 940, cloud_image_uploaded: false, api_calls: 1 },
};

describe("TracePanel", () => {
  it("reveals only supplied execution telemetry", () => {
    render(<TracePanel response={response} />);
    expect(screen.queryByText("Code detected")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Show Process" }));
    expect(screen.getByText("Code detected")).toBeInTheDocument();
    expect(screen.getByText("code · 94%")).toBeInTheDocument();
    expect(screen.getByText("1.20 sec")).toBeInTheDocument();
    expect(screen.getByText("Not uploaded")).toBeInTheDocument();
  });
});
