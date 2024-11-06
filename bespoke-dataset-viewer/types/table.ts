import { ReactNode } from 'react'

export type SortDirection = "asc" | "desc"

export interface Column {
  key: string
  label: string
}

export interface SortableHeaderProps {
  column: Column
  onSort: (columnKey: string) => void
  sortColumn: string | null
  sortDirection: SortDirection
  onFilter: (columnKey: string, value: string) => void
  filterValue: string
}

export interface SortableTableProps {
  columns: Column[]
  data: any[]
  getRowKey: (row: any) => string | number
  getCellContent: (row: any, columnKey: string) => ReactNode
  onRowClick?: (row: any) => void
  initialSortColumn?: string
  initialSortDirection?: SortDirection
} 