"use client"

import { useState, useCallback, useEffect, useMemo } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Table,
  TableBody,
  TableCell,
  TableHeader,
  TableRow,
  TableHead,
} from "@/components/ui/table"
import { Moon, Sun } from "lucide-react"
import { DndContext, closestCenter, KeyboardSensor, PointerSensor, useSensor, useSensors } from '@dnd-kit/core'
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy, arrayMove } from '@dnd-kit/sortable'
import { DataItem } from "@/types/dataset"
import { getColumnValue } from "@/lib/utils"
import { SortableHeader } from "./SortableHeader"
import { DetailsSidebar } from "./DetailsSidebar"
import { DistributionChart } from "./DistributionChart"
import { MessagePreviewCell } from "./MessagePreviewCell"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu"

const TABLE_COLUMNS = [
  "User Message",
  "Assistant Message",
  "Prompt Tokens",
  "Completion Tokens",
]

const GROUPABLE_COLUMNS = [
  "Model",
  "System Message"
]

export function DatasetViewer() {
  const [data, setData] = useState<DataItem[]>([])
  const [sortColumn, setSortColumn] = useState<string | null>(null)
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("asc")
  const [filters, setFilters] = useState<Record<string, string>>({})
  const [groupBy, setGroupBy] = useState<string | null>(null)
  const [theme, setTheme] = useState<"light" | "dark">("light")
  const [mounted, setMounted] = useState(false)
  const [columns, setColumns] = useState(TABLE_COLUMNS)
  const [selectedDistribution, setSelectedDistribution] = useState<string | null>(null)
  const [selectedItem, setSelectedItem] = useState<DataItem | null>(null)

  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  )

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

  const handleFileUpload = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    try {
      const content = await new Promise<string>((resolve, reject) => {
        const reader = new FileReader()
        reader.onload = (e) => resolve(e.target?.result as string)
        reader.onerror = reject
        reader.readAsText(file)
      })

      const lines = content.split('\n').filter(line => line.trim() !== '')
      const jsonData = lines.map(line => JSON.parse(line))
      setData(jsonData)
    } catch (error) {
      console.error("Error parsing file:", error)
      alert("Failed to parse file. Please ensure it's a valid JSON file.")
    }
  }, [])

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

  const handleDragEnd = useCallback((event: any) => {
    const { active, over } = event
    if (active.id !== over.id) {
      setColumns((items) => {
        const oldIndex = items.indexOf(active.id)
        const newIndex = items.indexOf(over.id)
        return arrayMove(items, oldIndex, newIndex)
      })
    }
  }, [])

  const filteredData = useMemo(() => {
    return data.filter((item) => {
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

  // Don't render anything until mounted
  if (!mounted) {
    return null
  }

  return (
    <div className="min-h-screen bg-background text-foreground">
      <header className="border-b">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Image
              src="https://hebbkx1anhila5yf.public.blob.vercel-storage.com/image-sN2O0LK0cVw6NesKNPlJCoWAu7xfOm.png"
              alt="Bespoke Logo"
              width={32}
              height={32}
              className="object-contain"
            />
            <h1 className="text-2xl font-bold">Bespoke Dataset Viewer</h1>
          </div>
          <Button variant="ghost" size="icon" onClick={toggleTheme}>
            {theme === 'light' ? (
              <Moon className="h-5 w-5" />
            ) : (
              <Sun className="h-5 w-5" />
            )}
            <span className="sr-only">Toggle theme</span>
          </Button>
        </div>
      </header>

      <main className="container mx-auto p-4">
        <Input
          type="file"
          accept=".json"
          onChange={handleFileUpload}
          className="mb-4"
        />
        
        {data.length > 0 && (
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
                  <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <SortableContext items={columns} strategy={verticalListSortingStrategy}>
                            {columns.map((column) => (
                              <SortableHeader 
                                key={column} 
                                column={column} 
                                onSort={handleSort}
                                sortColumn={sortColumn}
                                sortDirection={sortDirection}
                                onFilter={handleFilter}
                                filterValue={filters[column] || ""}
                              />
                            ))}
                          </SortableContext>
                          <TableHead>View Details</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {items.map((item, index) => (
                          <TableRow key={index}>
                            {columns.map((column) => (
                              <MessagePreviewCell
                                key={column}
                                content={getColumnValue(item, column)}
                                column={column}
                              />
                            ))}
                            <TableCell>
                              <Button variant="outline" onClick={() => setSelectedItem(item)}>
                                View Details
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </DndContext>
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