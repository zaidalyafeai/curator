import { DatasetViewer } from "@/components/dataset-viewer/DatasetViewer"

export default async function DatasetPage({ 
  params 
}: { 
  params: Promise<{ runHash: string }> 
}) {
  const { runHash } = await params
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <DatasetViewer runHash={runHash} />
      </body>
    </html>
  )
}
