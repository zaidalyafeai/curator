import type { Metadata } from "next";
import "./globals.css";


export const metadata: Metadata = {
  title: "Bella Dataset Viewer",
  description: "A powerful dataset viewer and analysis tool",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body suppressHydrationWarning>
        {children}
      </body>
    </html>
  )
}
