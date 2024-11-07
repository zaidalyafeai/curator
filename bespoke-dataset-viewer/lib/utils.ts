import { DataItem } from "../types/dataset"
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export const getColumnValue = (item: DataItem, column: string): string => {
  if (!item) return "N/A"

  switch (column) {
    case "Model":
      return item.request.model || "N/A"
    case "System Message":
      return item.request.messages.find((m: { role: string }) => m.role === "system")?.content || "N/A"
    case "User Message":
      return item.request.messages.find((m: { role: string }) => m.role === "user")?.content || "N/A"
    case "Assistant Message":
      return item.raw_response.choices[0]?.message?.content || "N/A"
    case "Total Tokens":
      return item.raw_response.usage.total_tokens.toString() || "N/A"
    case "Prompt Tokens":
      return item.raw_response.usage.prompt_tokens.toString() || "N/A"
    case "Completion Tokens":
      return item.raw_response.usage.completion_tokens.toString() || "N/A"
    default:
      return "N/A"
  }
}

export function getDistributionData(data: DataItem[], column: string) {
  // Extract values from the nested structure
  const values = data.map(item => {
    switch (column) {
      case "total_tokens":
        return item.raw_response.usage.total_tokens;
      case "prompt_tokens":
        return item.raw_response.usage.prompt_tokens;
      case "completion_tokens":
        return item.raw_response.usage.completion_tokens;
      default:
        return null;
    }
  }).filter((value): value is number => value !== null);

  if (values.length === 0) return [];

  // Get unique values
  const uniqueValues = [...new Set(values)];
  
  // If there's only one unique value, return it directly
  if (uniqueValues.length === 1) {
    return [{
      range: uniqueValues[0].toString(),
      start: uniqueValues[0],
      end: uniqueValues[0],
      count: values.length
    }];
  }

  // Get unique values and check if they're all integers
  const allIntegers = values.every(v => Number.isInteger(v));

  let min = Math.min(...values);
  let max = Math.max(...values);

  // Calculate data range and analyze distribution
  const range = max - min;

  // Determine appropriate bin size based on data characteristics
  let binSize;
  if (allIntegers && range <= 20) {
    // For small ranges of integers, use 1 as bin size
    binSize = 1;
  } else if (range <= 100) {
    // For small ranges, use approximately 10-15 bins
    binSize = Math.ceil(range / 10);
  } else {
    // For larger ranges, use Freedman-Diaconis rule with some adjustments
    const iqr = calculateIQR(values);
    binSize = 2 * iqr * Math.pow(values.length, -1/3);
    
    // Round to a nice number
    const magnitude = Math.pow(10, Math.floor(Math.log10(binSize)));
    binSize = Math.ceil(binSize / magnitude) * magnitude;
  }

  // Adjust min and max to create nice boundaries
  min = Math.floor(min / binSize) * binSize;
  max = Math.ceil(max / binSize) * binSize;

  // Create bins
  const bins = [];
  let start = min;
  while (start < max) {
    const end = start + binSize;
    bins.push({
      range: allIntegers 
        ? `${Math.floor(start)}-${Math.floor(end)}`
        : `${start.toFixed(2)}-${end.toFixed(2)}`,
      start,
      end,
      count: values.filter(v => v >= start && v < end).length
    });
    start = end;
  }

  // Handle last bin edge case to include max value
  if (bins.length > 0) {
    const lastBin = bins[bins.length - 1];
    lastBin.count += values.filter(v => v === lastBin.end).length;
  }

  return bins;
}

// Helper function to calculate Interquartile Range
function calculateIQR(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = sorted[Math.floor(sorted.length * 0.25)];
  const q3 = sorted[Math.floor(sorted.length * 0.75)];
  return q3 - q1;
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}