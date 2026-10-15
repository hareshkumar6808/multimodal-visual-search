import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { MarkdownAnswer } from "./MarkdownAnswer";

describe("MarkdownAnswer", () => {
  it("renders structured assistant text as headings, lists, and code", () => {
    const { container } = render(
      <MarkdownAnswer>{"## Explanation\n\n- First point\n- Second point\n\n`x = 2`"}</MarkdownAnswer>,
    );
    expect(screen.getByRole("heading", { name: "Explanation" })).toBeInTheDocument();
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
    expect(container.querySelector("code")).toHaveTextContent("x = 2");
  });
});
