import "./globals.css";
import "leaflet/dist/leaflet.css";
import type { Metadata } from "next";
import type { ReactNode } from "react";

export const metadata: Metadata = {
  title: "Multi Agent Travel Planner",
  description: "Next.js frontend for the FastAPI travel planning backend.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
