import { DatasetViewer } from "@/components/dataset-viewer/DatasetViewer"

export default function DatasetPage({ params }: { params: { runHash: string } }) {
  return <DatasetViewer runHash={params.runHash} />
} 