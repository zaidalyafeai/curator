"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { SortableTable } from "@/components/ui/sortable-table"
import { Column } from "@/types/table"
import { Run } from "@/types/dataset"

const COLUMNS: Column[] = [
  { key: "timestamp", label: "Timestamp" },
  { key: "model_name", label: "Model" },
  { key: "system_prompt", label: "System Prompt" },
  { key: "user_prompt", label: "User Prompt" },
  { key: "response_format", label: "Response Format" }
]

export function RunsTable() {
  const [runs, setRuns] = useState<Run[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  useEffect(() => {
    const fetchRuns = async () => {
      try {
        setIsLoading(true)
        setError(null)
        const response = await fetch('/api/runs')
        const data = await response.json()
        
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`)
        }

        if (!Array.isArray(data)) {
          throw new Error('Data is not in the expected format')
        }
        
        setRuns(data)
      } catch (error) {
        console.error('Failed to fetch runs:', error)
        setError(error instanceof Error ? error.message : 'Failed to fetch runs')
      } finally {
        setIsLoading(false)
      }
    }

    fetchRuns()
  }, [])

  const getCellContent = (run: Run, columnKey: string) => {
    switch (columnKey) {
      case "timestamp":
        return new Date(run.timestamp).toLocaleString()
      default:
        return run[columnKey as keyof Run]
    }
  }

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>
  if (runs.length === 0) return <div>No runs found</div>

  return (
    <SortableTable
      columns={COLUMNS}
      data={runs}
      getRowKey={(run) => run.id}
      getCellContent={getCellContent}
      onRowClick={(run) => router.push(`/dataset/${run.run_hash}`)}
      truncateConfig={{ 
        enabled: true, 
        maxLength: 100 // Adjust based on your needs
      }}
    />
  )
}