"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { DataItem } from '@/types/dataset'
import { getDistributionData } from '@/lib/utils'

interface DistributionChartProps {
  data: DataItem[]
  column: string
}

// Move this outside component
const COLUMN_DISPLAY_NAMES: Record<string, string> = {
  total_tokens: "Total Tokens",
  prompt_tokens: "Prompt Tokens",
  completion_tokens: "Completion Tokens"
}

export function DistributionChart({ data, column }: DistributionChartProps) {
  const distributionData = getDistributionData(data, column);

  if (distributionData.length === 0) return null;

  const displayName = COLUMN_DISPLAY_NAMES[column] || column;

  return (
    <div className="mb-8 h-64">
      <h2 className="text-xl font-semibold mb-2">Distribution of {displayName}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={distributionData}>
          <XAxis 
            dataKey="range" 
            fontSize={12}
            angle={-45}
            textAnchor="end"
            height={60}
          />
          <YAxis />
          <Tooltip 
            formatter={(value: number) => [`Count: ${value}`, displayName]}
          />
          <Bar 
            dataKey="count" 
            fill="#8884d8"
            name={displayName}
          />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}