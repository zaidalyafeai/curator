"use client"

import { useEffect, useState, useCallback } from "react"
import { useRouter } from "next/navigation"
import { SortableTable } from "@/components/ui/sortable-table"
import { Column } from "@/types/table"
import { Run } from "@/types/dataset"
import { useToast } from "@/components/ui/use-toast"
import { Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const COLUMNS: Column[] = [
  { key: "created_time", label: "Created" },
  { key: "last_edited_time", label: "Last Updated" },
  { key: "model_name", label: "Model" },
  { key: "prompt_func", label: "Prompter Function" },
  { key: "response_format", label: "Response Format" }
]

export function RunsTable() {
  const [runs, setRuns] = useState<Run[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isPolling, setIsPolling] = useState(true)
  const [lastCreatedTime, setLastCreatedTime] = useState<string | null>(null)
  const [newRunIds, setNewRunIds] = useState<Set<string>>(new Set())
  const router = useRouter()
  const { toast } = useToast()

  const fetchRuns = useCallback(async (isInitial = false) => {
    try {
      const queryParams = lastCreatedTime && !isInitial 
        ? `?lastCreatedTime=${lastCreatedTime}`
        : ''
      
      const response = await fetch(`/api/runs${queryParams}`)
      const data = await response.json()
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      if (!Array.isArray(data)) {
        throw new Error('Data is not in the expected format')
      }

      if (data.length > 0) {
        if (isInitial) {
          setRuns(data)
          setLastCreatedTime(data[0].created_time)
        } else if (data.length > 0) {
          // Mark new runs for highlighting
          const newIds = new Set(data.map(run => run.run_hash))
          setNewRunIds(newIds)
          
          // Add new runs to the top
          setRuns(prevRuns => [...data, ...prevRuns])
          setLastCreatedTime(data[0].created_time)

          // Show toast for new runs
          toast({
            title: "New runs available",
            description: `${data.length} new runs added`,
          })

          // Clear highlighting after 5 seconds
          setTimeout(() => {
            setNewRunIds(new Set())
          }, 5000)
        }
      }
    } catch (error) {
      console.error('Failed to fetch runs:', error)
      setError(error instanceof Error ? error.message : 'Failed to fetch runs')
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch new runs",
      })
    } finally {
      setIsLoading(false)
    }
  }, [lastCreatedTime, toast])

  // Initial load
  useEffect(() => {
    fetchRuns(true)
  }, [fetchRuns])

  // Polling effect
  useEffect(() => {
    if (!isPolling) return

    const pollInterval = setInterval(() => {
      fetchRuns(false)
    }, 5000)

    return () => clearInterval(pollInterval)
  }, [isPolling, fetchRuns])

  const getCellContent = (run: Run, columnKey: string) => {
    switch (columnKey) {
      case "created_time":
        return new Date(run.created_time).toLocaleString()
      case "last_edited_time":
        return run.last_edited_time === '-' 
          ? '-' 
          : new Date(run.last_edited_time).toLocaleString()
      default:
        return run[columnKey as keyof Run]
    }
  }

  if (isLoading) return <div>Loading...</div>
  if (error) return <div>Error: {error}</div>
  if (runs.length === 0) return <div>No runs found</div>

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-semibold"/>
        <div className="flex items-center gap-4">
          {isPolling && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              <span>Polling for updates...</span>
            </div>
          )}
          <Button 
            variant="outline" 
            size="sm"
            onClick={() => setIsPolling(prev => !prev)}
          >
            {isPolling ? 'Stop' : 'Start'} Updates
          </Button>
        </div>
      </div>
      
      <SortableTable
        columns={COLUMNS}
        data={runs}
        getRowKey={(run) => run.id}
        getCellContent={getCellContent}
        onRowClick={(run) => router.push(`/dataset/${run.run_hash}`)}
        truncateConfig={{ enabled: true, maxLength: 100 }}
        pageSize={10}
        rowProps={(run) => ({
          className: cn(
            newRunIds.has(run.id) && "bg-success/30 animate-highlight",
            "transition-colors duration-300"
          )
        })}
      />
    </div>
  )
}