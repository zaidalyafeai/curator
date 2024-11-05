"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
  TableHead,
} from "@/components/ui/table"

interface Run {
  id: number
  timestamp: string
  dataset_hash: string
  user_prompt: string
  system_prompt: string
  model_name: string
  response_format: string
  run_hash: string
}

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
          throw new Error(`HTTP error! status: ${response.status}, message: ${data.error || 'Unknown error'}`)
        }

        // Ensure data is an array
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

  const handleRowClick = (runHash: string) => {
    router.push(`/dataset/${runHash}`)
  }

  if (isLoading) {
    return <div>Loading...</div>
  }

  if (error) {
    return <div>Error: {error}</div>
  }

  if (runs.length === 0) {
    return <div>No runs found</div>
  }

  return (
    <div className="rounded-lg border bg-card">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Timestamp</TableHead>
            <TableHead>Model</TableHead>
            <TableHead>User Prompt</TableHead>
            <TableHead>System Prompt</TableHead>
            <TableHead>Response Format</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {runs.map((run) => (
            <TableRow
              key={run.id}
              className="cursor-pointer hover:bg-muted/50"
              onClick={() => handleRowClick(run.run_hash)}
            >
              <TableCell>{new Date(run.timestamp).toLocaleString()}</TableCell>
              <TableCell>{run.model_name}</TableCell>
              <TableCell className="max-w-[300px] truncate">{run.user_prompt}</TableCell>
              <TableCell className="max-w-[300px] truncate">{run.system_prompt}</TableCell>
              <TableCell>{run.response_format}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
} 