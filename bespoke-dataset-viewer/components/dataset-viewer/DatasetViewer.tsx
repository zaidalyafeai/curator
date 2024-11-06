"use client"

import { useState, useCallback, useEffect, useMemo } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Moon, Sun, Loader2, Bell } from "lucide-react"
import { DataItem } from "@/types/dataset"
import { getColumnValue } from "@/lib/utils"
import { DetailsSidebar } from "./DetailsSidebar"
import { DistributionChart } from "./DistributionChart"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"
import { useRouter } from "next/navigation"
import { Column } from "@/types/table"
import { SortableTable } from "@/components/ui/sortable-table"
import { useToast } from "@/components/ui/use-toast"
import { motion, AnimatePresence } from "framer-motion"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

const COLUMNS: Column[] = [
  { key: "user_message", label: "User Message" },
  { key: "assistant_message", label: "Assistant Message" },
  { key: "prompt_tokens", label: "Prompt Tokens" },
  { key: "completion_tokens", label: "Completion Tokens" }
]

const GROUPABLE_COLUMNS = [
  "Model",
  "System Message"
]

interface DatasetViewerProps {
  runHash?: string
}

export function DatasetViewer({ runHash }: DatasetViewerProps) {
  const router = useRouter()
  const [data, setData] = useState<DataItem[]>([])
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc")
  const [filters, setFilters] = useState<Record<string, string>>({})
  const [groupBy, setGroupBy] = useState<string | null>(null)
  const [theme, setTheme] = useState<"light" | "dark">("light")
  const [mounted, setMounted] = useState(false)
  const [selectedDistribution, setSelectedDistribution] = useState<string | null>("total_tokens")
  const [selectedItem, setSelectedItem] = useState<DataItem | null>(null)
  const { toast } = useToast()
  const [isPolling, setIsPolling] = useState(false)
  const [lastLineNumber, setLastLineNumber] = useState(0)
  const [isInitialLoad, setIsInitialLoad] = useState(true)
  const [newItemIds, setNewItemIds] = useState<Set<string>>(new Set())
  const [hasNewData, setHasNewData] = useState(false)

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

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  const handleSort = useCallback((column: string) => {
    if (sortColumn === column) {
      setSortDirection(prev => prev === "asc" ? "desc" : "asc")
    } else {
      setSortColumn(column)
      setSortDirection("asc")
    }
  }, [sortColumn])

  const handleFilter = useCallback((column: string, value: string) => {
    setFilters(prev => ({
      ...prev,
      [column]: value
    }))
  }, [])

  const handleGroup = useCallback((column: string | null) => {
    setGroupBy(column)
  }, [])

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

  const groupedData = useMemo(() => {
    if (!groupBy) return { "": sortedData }
    
    return sortedData.reduce((groups, item) => {
      const groupKey = getColumnValue(item, groupBy)
      if (!groups[groupKey]) groups[groupKey] = []
      groups[groupKey].push(item)
      return groups
    }, {} as Record<string, DataItem[]>)
  }, [sortedData, groupBy])
  
  useEffect(() => {
  console.log('Raw data:', data);
  console.log('Filtered data:', filteredData);
  console.log('Sorted data:', sortedData);
  console.log('Grouped data:', groupedData);
  }, [data, filteredData, sortedData, groupedData])
  const getCellContent = (item: DataItem, columnKey: string) => {
    const [requestData, responseData] = item;
    
    switch (columnKey) {
      case "user_message":
        return requestData.messages.find(m => m.role === "user")?.content || "N/A";
      case "assistant_message":
        return responseData.choices[0]?.message?.content || "N/A";
      case "prompt_tokens":
        return responseData.usage.prompt_tokens?.toString() || "N/A";
      case "completion_tokens":
        return responseData.usage.completion_tokens?.toString() || "N/A";
      default:
        return "N/A";
    }
  }

  const fetchNewResponses = useCallback(async () => {
    if (!runHash) return

    try {
      const response = await fetch(`/api/responses/${runHash}?lastLine=${lastLineNumber}`)
      if (!response.ok) throw new Error('Failed to fetch responses')
      
      const { data: newData, totalLines } = await response.json()
      
      if (newData && newData.length > 0) {
        setNewItemIds(new Set(newData.map((item: DataItem) => item.id)))
        setHasNewData(true)

        setData(prevData => [...newData.reverse(), ...prevData])
        setLastLineNumber(totalLines)
        
        setTimeout(() => {
          setNewItemIds(new Set())
          setHasNewData(false)
        }, 5000)

        toast({
          title: "New responses received",
          description: `${newData.length} new responses added`,
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
  }, [runHash, lastLineNumber, toast])

  const handleInitialLoad = useCallback(async () => {
    if (!runHash) return

    try {
      const response = await fetch(`/api/responses/${runHash}`)
      if (!response.ok) throw new Error('Failed to fetch responses')
      const { data: initialData, totalLines } = await response.json()
      
      if (initialData && Array.isArray(initialData)) {
        setData(initialData.reverse()) // Newest first
        setLastLineNumber(totalLines)
        
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
  }, [runHash, toast])

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
    <div className="min-h-screen bg-background text-foreground overflow-x-hidden">
      <header className="border-b sticky top-0 z-50 bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/75">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2 cursor-pointer" onClick={() => router.push('/')}>
            <Image
              src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-sN2O0LK0cVw6NesKNPlJCoWAu7xfOm.png"
              alt="Bespoke Logo"
              width={32}
              height={32}
              className="object-contain"
            />
            <h1 className="text-2xl font-bold">Bespoke Dataset Viewer</h1>
          </div>
          <div className="flex items-center gap-4">
            {isInitialLoad ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Loading dataset...</span>
              </div>
            ) : isPolling ? (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span>Polling for updates...</span>
              </div>
            ) : null}
            <Button 
              variant="outline" 
              size="sm"
              onClick={() => setIsPolling(prev => !prev)}
              disabled={isInitialLoad}
            >
              {isPolling ? 'Stop' : 'Start'} Updates
            </Button>
            <Button variant="ghost" size="icon" onClick={toggleTheme}>
              {theme === 'light' ? (
                <Moon className="h-5 w-5" />
              ) : (
                <Sun className="h-5 w-5" />
              )}
              <span className="sr-only">Toggle theme</span>
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto p-4 overflow-x-hidden">
        {isInitialLoad ? (
          <div className="flex items-center justify-center h-64">
            <div className="flex flex-col items-center gap-4">
              <Loader2 className="h-8 w-8 animate-spin" />
              <p className="text-muted-foreground">Loading dataset...</p>
            </div>
          </div>
        ) : (
          <>
            <div className="mb-4 flex space-x-2">
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline">Group By</Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
                  <DropdownMenuItem onClick={() => handleGroup(null)}>
                    None
                  </DropdownMenuItem>
                  {GROUPABLE_COLUMNS.map((column) => (
                    <DropdownMenuItem key={column} onClick={() => handleGroup(column)}>
                      {column}
                    </DropdownMenuItem>
                  ))}
                </DropdownMenuContent>
              </DropdownMenu>

              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button variant="outline">Show Distribution</Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent>
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

            {selectedDistribution && (
              <DistributionChart
                data={sortedData}
                column={selectedDistribution}
              />
            )}

            {Object.entries(groupedData).map(([group, items]) => (
              <div key={group} className="mb-8">
                {groupBy && <h2 className="text-xl font-semibold mb-2">{group}</h2>}
                <div className="rounded-lg border bg-card">
                  <AnimatePresence>
                    <SortableTable
                      columns={COLUMNS}
                      data={items}
                      getRowKey={(item) => item.id}
                      getCellContent={getCellContent}
                      onRowClick={(item) => setSelectedItem(item)}
                      truncateConfig={{ 
                        enabled: true, 
                        maxLength: 150
                      }}
                      rowProps={(item) => ({
                        className: cn(
                          newItemIds.has(item.id) && "bg-success/30 animate-highlight",
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
              </div>
            ))}
          </>
        )}
      </main>
      <DetailsSidebar item={selectedItem} onClose={() => setSelectedItem(null)} />
    </div>
  )
}