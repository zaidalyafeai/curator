"use client"

import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import { DataItem } from '@/types/dataset'

interface DistributionChartProps {
  data: DataItem[]
  column: string
}

const COLUMN_DISPLAY_NAMES: Record<string, string> = {
  total_tokens: "Total Tokens",
  prompt_tokens: "Prompt Tokens",
  completion_tokens: "Completion Tokens"
}

function calculateNiceBinSize(min: number, max: number): { binSize: number, start: number, end: number } {
  // Calculate the range
  const range = max - min;
  
  // Get the magnitude of the range (in powers of 10)
  const magnitude = Math.floor(Math.log10(range));
  
  // Possible bin sizes (1, 2, 5, 10) * 10^n
  const base = Math.pow(10, magnitude);
  const possibleBinSizes = [
    0.1 * base,
    0.2 * base,
    0.5 * base,
    base,
    2 * base,
    5 * base,
    10 * base
  ];

  // Aim for between 5-10 bins
  const targetBins = 7;
  
  // Find the bin size that gives us closest to our target number of bins
  const binSize = possibleBinSizes.reduce((prev, curr) => {
    const prevBins = range / prev;
    const currBins = range / curr;
    return Math.abs(currBins - targetBins) < Math.abs(prevBins - targetBins) ? curr : prev;
  });

  // Calculate nice start and end points
  const start = Math.floor(min / binSize) * binSize;
  const end = Math.ceil(max / binSize) * binSize;

  return { binSize, start, end };
}

function getDistributionData(data: DataItem[], column: string) {
  const values = data.map(item => {
    switch (column) {
      case "total_tokens":
        return item.raw_response.usage.total_tokens;
      case "prompt_tokens":
        return item.raw_response.usage.prompt_tokens;
      case "completion_tokens":
        return item.raw_response.usage.completion_tokens;
      default:
        return 0;
    }
  });

  const min = Math.min(...values);
  const max = Math.max(...values);

  // If min equals max, all values are the same
  if (min === max) {
    return [{
      range: `${min}`,
      count: values.length
    }];
  }

  const { binSize, start, end } = calculateNiceBinSize(min, max);
  const numBins = Math.ceil((end - start) / binSize);
  
  // Initialize buckets
  const buckets: { range: string; start: number; count: number }[] = [];
  for (let i = 0; i < numBins; i++) {
    const bucketStart = start + (i * binSize);
    const bucketEnd = bucketStart + binSize;
    buckets.push({
      range: `${Math.round(bucketStart)}-${Math.round(bucketEnd)}`,
      start: bucketStart,
      count: 0
    });
  }

  // Count values in buckets
  values.forEach(value => {
    const bucketIndex = Math.floor((value - start) / binSize);
    if (bucketIndex >= 0 && bucketIndex < buckets.length) {
      buckets[bucketIndex].count++;
    }
  });

  // Only return buckets that have counts
  return buckets
    .filter(bucket => bucket.count > 0)
    .sort((a, b) => a.start - b.start)
    .map(({ range, count }) => ({
      range,
      count
    }));
}

export function DistributionChart({ data, column }: DistributionChartProps) {
  const distributionData = getDistributionData(data, column);
  
  if (distributionData.length === 0) return null;

  const displayName = COLUMN_DISPLAY_NAMES[column] || column;

  return (
    <div className="w-full h-80">
      <h2 className="text-xl font-semibold mb-4">Distribution of {displayName}</h2>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart 
          data={distributionData}
          margin={{ top: 10, right: 30, left: 20, bottom: 15 }}
        >
          <XAxis 
            dataKey="range" 
            fontSize={12}
            angle={-45}
            textAnchor="end"
            height={70}
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