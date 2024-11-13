"use client"

import { Header } from "@/components/layout/Header"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { SortableTable } from "@/components/ui/sortable-table"
import { useToast } from "@/components/ui/use-toast"
import { cn, getColumnValue } from "@/lib/utils"
import { DataItem } from "@/types/dataset"
import { Column } from "@/types/table"
import { AnimatePresence } from "framer-motion"
import { Loader2 } from "lucide-react"
import { useCallback, useEffect, useMemo, useState } from "react"
import { DetailsSidebar } from "./DetailsSidebar"
import { DistributionChart } from "./DistributionChart"

const COLUMNS: Column[] = [
  { key: "user_message", label: "User Message" },
  { key: "assistant_message", label: "Assistant Message" },
  { key: "prompt_tokens", label: "Prompt Tokens" },
  { key: "completion_tokens", label: "Completion Tokens" }
]

interface DatasetViewerProps {
  runHash?: string
  batchMode: boolean
}

export function DatasetViewer({ runHash, batchMode }: DatasetViewerProps) {
  const [data, setData] = useState<DataItem[]>([])
  const [sortColumn] = useState<string | null>(null)
  const [sortDirection] = useState<"asc" | "desc">("asc")
  const [filters] = useState<Record<string, string>>({})
  const [theme, setTheme] = useState<"light" | "dark">("light")
  const [mounted, setMounted] = useState(false)
  const [selectedDistribution, setSelectedDistribution] = useState<string | null>("total_tokens")
  const [selectedItem, setSelectedItem] = useState<DataItem | null>(null)
  const { toast } = useToast()
  const [isPolling, setIsPolling] = useState(false)
  const [lastLineNumber, setLastLineNumber] = useState(0)
  const [isInitialLoad, setIsInitialLoad] = useState(true)
  const [newItemIds, setNewItemIds] = useState<Set<number>>(new Set())
  const [processedFiles, setProcessedFiles] = useState<string[]>([])

  useEffect(() => {
    const systemPreference = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null
    setTheme(savedTheme || systemPreference)
    setMounted(true)
  }, [])

  useEffect(() => {
    if (!mounted) return

    if (theme === 'dark') {
      document.documentElement.classList.add('dark')
    } else {
      document.documentElement.classList.remove('dark')
    }
    localStorage.setItem('theme', theme)
  }, [theme, mounted])

  const filteredData = useMemo(() => {
    const dataArray = Array.isArray(data) ? data : []

    return dataArray.filter((item) => {
      return Object.entries(filters).every(([column, filterValue]) => {
        if (!filterValue) return true
        const cellValue = getColumnValue(item, column)
        return cellValue.toLowerCase().includes(filterValue.toLowerCase())
      })
    })
  }, [data, filters])

  const sortedData = useMemo(() => {
    if (!sortColumn) return filteredData

    return [...filteredData].sort((a, b) => {
      const aValue = getColumnValue(a, sortColumn)
      const bValue = getColumnValue(b, sortColumn)

      const comparison = aValue.localeCompare(bValue)
      return sortDirection === "asc" ? comparison : -comparison
    })
  }, [filteredData, sortColumn, sortDirection])

  const fetchNewResponses = useCallback(async () => {
    if (!runHash) return

    try {
      const queryParams = new URLSearchParams({
        batchMode: batchMode.toString(),
        ...(batchMode
          ? { processedFiles: processedFiles.join(',') }
          : { lastLine: lastLineNumber.toString() }
        )
      })

      const response = await fetch(`/api/responses/${runHash}?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch responses')

      const responseData = await response.json()

      if (responseData.data && responseData.data.length > 0) {
        setNewItemIds(new Set(responseData.data.map((item: DataItem) => item.raw_response.id)))

        setData(prevData => [...responseData.data.reverse(), ...prevData])

        if (batchMode) {
          // Update processed files list
          setProcessedFiles(prev => [...prev, ...responseData.processedFiles])
        } else {
          // Update last line number for streaming mode
          setLastLineNumber(responseData.totalLines)
        }

        setTimeout(() => {
          setNewItemIds(new Set())
        }, 5000)

        toast({
          title: "New responses received",
          description: `${responseData.data.length} new responses added`,
        })
      }
    } catch (error) {
      console.error("Error fetching responses:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to fetch new responses",
      })
    }
  }, [runHash, lastLineNumber, processedFiles, batchMode, toast])

  const handleInitialLoad = useCallback(async () => {
    if (!runHash) return

    try {
      const queryParams = new URLSearchParams({
        batchMode: batchMode.toString(),
        ...(batchMode
          ? { processedFiles: '' }
          : { lastLine: '0' }
        )
      })
      const response = await fetch(`/api/responses/${runHash}?${queryParams}`)
      if (!response.ok) throw new Error('Failed to fetch responses')
      const { data: initialData, totalLines, processedFiles: newProcessedFiles } = await response.json()

      if (initialData && Array.isArray(initialData)) {
        setData(initialData.reverse()) // Newest first

        if (batchMode && newProcessedFiles) {
          setProcessedFiles(newProcessedFiles)
        } else {
          setLastLineNumber(totalLines)
        }

        // Show initial data count
        toast({
          title: "Dataset loaded",
          description: `Loaded ${initialData.length} responses`,
        })
      }

      // Start polling after initial load
      setIsPolling(true)
    } catch (error) {
      console.error("Error reading responses file:", error)
      toast({
        variant: "destructive",
        title: "Error",
        description: "Failed to load dataset",
      })
    } finally {
      setIsInitialLoad(false)
    }
  }, [runHash, batchMode, toast])

  // Initial data load
  useEffect(() => {
    if (runHash && isInitialLoad) {
      handleInitialLoad()
    }
  }, [runHash, isInitialLoad, handleInitialLoad])

  // Modify the polling effect to avoid race conditions
  useEffect(() => {
    if (!isPolling || !runHash || isInitialLoad) return

    const pollInterval = setInterval(fetchNewResponses, 5000)

    return () => {
      clearInterval(pollInterval)
    }
  }, [isPolling, runHash, fetchNewResponses, isInitialLoad])

  // Don't render anything until mounted
  if (!mounted) {
    return null
  }

  return (
    <>
      <DetailsSidebar item={selectedItem} onClose={() => setSelectedItem(null)} />
      <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
        <Header
          isLoading={isInitialLoad}
          isPolling={isPolling}
          onTogglePolling={() => setIsPolling(prev => !prev)}
          loadingText="Loading dataset..."
          pollingText="Polling for updates..."
        />

        <main className="container mx-auto p-4 overflow-x-hidden">
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h2 className="text-2xl font-semibold text-foreground">Dataset Details</h2>
              <p className="text-sm text-muted-foreground">View and analyze your dataset responses</p>
            </div>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline">
                  {selectedDistribution
                    ? selectedDistribution.split('_').map(word =>
                      word.charAt(0).toUpperCase() + word.slice(1)
                    ).join(' ')
                    : 'Select Metric'}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={() => setSelectedDistribution(null)}>
                  None
                </DropdownMenuItem>
                {["total_tokens", "prompt_tokens", "completion_tokens"].map((column) => (
                  <DropdownMenuItem key={column} onClick={() => setSelectedDistribution(column)}>
                    {column === "total_tokens" ? "Total Tokens" :
                      column === "prompt_tokens" ? "Prompt Tokens" :
                        "Completion Tokens"}
                  </DropdownMenuItem>
                ))}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {isInitialLoad ? (
            <div className="flex items-center justify-center h-64">
              <div className="flex flex-col items-center gap-4">
                <Loader2 className="h-8 w-8 animate-spin" />
                <p className="text-muted-foreground">Loading dataset...</p>
              </div>
            </div>
          ) : (
            <>
              <div className="mb-8 space-y-4">
                {selectedDistribution && (
                  <div className="rounded-lg border bg-card p-4">
                    <DistributionChart
                      data={sortedData}
                      column={selectedDistribution}
                    />
                  </div>
                )}
              </div>

              <div className="rounded-lg border bg-card">
                <AnimatePresence>
                  <SortableTable
                    columns={COLUMNS}
                    data={sortedData}
                    getRowKey={(item) => item.raw_response.id}
                    getCellContent={(item, columnKey) => getColumnValue(item, columnKey)}
                    onRowClick={(item) => setSelectedItem(item)}
                    truncateConfig={{
                      enabled: true,
                      maxLength: 150
                    }}
                    pageSize={10}
                    rowProps={(item) => ({
                      className: cn(
                        newItemIds.has(item.raw_response.id) && "bg-success/30 animate-highlight",
                        "transition-colors duration-300"
                      ),
                      layout: true,
                      initial: { opacity: 0, y: -20 },
                      animate: {
                        opacity: 1,
                        y: 0,
                        transition: {
                          duration: 0.2
                        }
                      },
                      exit: {
                        opacity: 0,
                        y: -20,
                        transition: {
                          duration: 0.2
                        }
                      },
                    })}
                  />
                </AnimatePresence>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  )
}