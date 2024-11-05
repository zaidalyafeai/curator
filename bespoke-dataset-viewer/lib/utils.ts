import { DataItem } from "../types/dataset"
import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export const getColumnValue = (item: DataItem, column: string): string => {
  if (!item || item.length < 2) return "N/A"

  const [requestData, responseData] = item

  switch (column) {
    case "Model":
      return requestData.model || "N/A"
    case "System Message":
      return requestData.messages.find(m => m.role === "system")?.content || "N/A"
    case "User Message":
      return requestData.messages.find(m => m.role === "user")?.content || "N/A"
    case "Assistant Message":
      return responseData.choices[0]?.message?.content || "N/A"
    case "Total Tokens":
      return responseData.usage.total_tokens.toString() || "N/A"
    case "Prompt Tokens":
      return responseData.usage.prompt_tokens.toString() || "N/A"
    case "Completion Tokens":
      return responseData.usage.completion_tokens.toString() || "N/A"
    default:
      return "N/A"
  }
}

export function getDistributionData(data: DataItem[], column: string) {
  // Extract token values from the nested structure
  const values = data.map(item => {
    const response = item[1];
    if (!response?.usage) return null;
    
    switch (column) {
      case "total_tokens":
        return response.usage.total_tokens;
      case "prompt_tokens":
        return response.usage.prompt_tokens;
      case "completion_tokens":
        return response.usage.completion_tokens;
      default:
        return null;
    }
  }).filter((value): value is number => value !== null);

  if (values.length === 0) return [];

  // Create distribution bins
  const min = Math.min(...values);
  const max = Math.max(...values);
  
  // Handle case where all values are the same
  if (min === max) {
    // Create a range around the single value
    const padding = Math.ceil(min * 0.1); // Add 10% padding
    return Array.from({ length: 3 }, (_, i) => ({
      range: `${min + (i - 1) * padding}-${min + i * padding}`,
      start: min + (i - 1) * padding,
      end: min + i * padding,
      count: i === 1 ? values.length : 0 // Only middle bin has the count
    }));
  }

  const binCount = 10;
  // Round bin size to a nice number for better readability
  let binSize = (max - min) / binCount;
  const magnitude = Math.pow(10, Math.floor(Math.log10(binSize)));
  binSize = Math.ceil(binSize / magnitude) * magnitude;

  // Adjust min and max to ensure even distribution
  const adjustedMin = Math.floor(min / binSize) * binSize;
  const adjustedMax = Math.ceil(max / binSize) * binSize;
  const adjustedBinCount = Math.ceil((adjustedMax - adjustedMin) / binSize);

  // Initialize bins with proper ranges
  const bins = Array.from({ length: adjustedBinCount }, (_, i) => {
    const start = adjustedMin + (i * binSize);
    const end = start + binSize;
    return {
      range: `${start}-${end}`,
      start,
      end,
      count: 0
    };
  });

  // Count values in each bin
  values.forEach(value => {
    const binIndex = Math.floor((value - adjustedMin) / binSize);
    if (binIndex >= 0 && binIndex < bins.length) {
      bins[binIndex].count++;
    }
  });

  // Return all bins to show gaps
  return bins;
}

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}