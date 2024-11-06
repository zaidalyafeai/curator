import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import { homedir } from 'os'
import { join } from 'path'

export async function GET(
  request: Request,
  { params }: { params: { hash: string } }
) {
  try {
    const hash = await params.hash
    const responsesPath = join(homedir(), '.cache', 'bella', hash, 'responses.jsonl')
    const content = await fs.readFile(responsesPath, 'utf-8')
    const lines = content.split('\n').filter(line => line.trim() !== '')
    const jsonData = lines.map(line => JSON.parse(line))
    return NextResponse.json(jsonData)
  } catch (error) {
    console.error("Error reading responses file:", error)
    return NextResponse.json(
      { error: "Failed to read responses file" },
      { status: 500 }
    )
  }
} 