import { RunsTable } from "@/components/dataset-viewer/RunsTable"

export default function Home() {
  return (
    <div className="container mx-auto p-4">
      <h1 className="text-2xl font-bold mb-4">Bespoke Dataset Runs</h1>
      <RunsTable />
    </div>
  )
}