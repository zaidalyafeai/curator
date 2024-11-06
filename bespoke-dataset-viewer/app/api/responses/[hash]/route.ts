import { NextResponse } from 'next/server'
import { promises as fs } from 'fs'
import { homedir } from 'os'
import { join } from 'path'

export const dynamic = 'force-dynamic'
export const runtime = 'nodejs'

export async function GET(
  request: Request,
  { params }: { params: { hash: string } }
) {
  try {
    // In Next.js 14, params.hash is already a string, no need to await it
    const hash = params.hash
    const responsesPath = join(homedir(), '.cache', 'bella', hash, 'responses.jsonl')
    
    // Get the last line number from query params
    const { searchParams } = new URL(request.url)
    const lastLineNumber = parseInt(searchParams.get('lastLine') || '0')

    const content = await fs.readFile(responsesPath, 'utf-8')
    const lines = content.split('\n').filter(line => line.trim() !== '')
    
    // Only return new lines after lastLineNumber
    const newLines = lines.slice(lastLineNumber)
    const jsonData = newLines.map(line => JSON.parse(line))

    return NextResponse.json({
      data: jsonData,
      totalLines: lines.length
    })
  } catch (error) {
    console.error("Error reading responses file:", error)
    return NextResponse.json(
      { error: "Failed to read responses file" },
      { status: 500 }
    )
  }
} 