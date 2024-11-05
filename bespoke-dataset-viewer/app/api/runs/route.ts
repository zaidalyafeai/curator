import { NextResponse } from 'next/server'
import { Database } from 'sqlite3'
import { homedir } from 'os'
import { join } from 'path'
import { existsSync } from 'fs'

export async function GET() {
  return new Promise((resolve) => {
    const dbPath = join(homedir(), '.cache', 'bella', 'metadata.db')
    
    // Check if database file exists
    if (!existsSync(dbPath)) {
      console.error(`Database file not found at: ${dbPath}`)
      resolve(NextResponse.json({ error: 'Database file not found' }, { status: 500 }))
      return
    }

    try {
      const db = new Database(dbPath, (err) => {
        if (err) {
          console.error('Database connection error:', err)
          resolve(NextResponse.json({ error: 'Database connection failed' }, { status: 500 }))
          return
        }
      })

      db.all(
        'SELECT * FROM runs ORDER BY timestamp DESC',
        (err, rows) => {
          if (err) {
            console.error('Database query error:', err)
            resolve(NextResponse.json({ error: 'Database query failed' }, { status: 500 }))
            return
          }
          
          const safeRows = Array.isArray(rows) ? rows : []
          resolve(NextResponse.json(safeRows))
          
          db.close((closeErr) => {
            if (closeErr) {
              console.error('Error closing database:', closeErr)
            }
          })
        }
      )
    } catch (error) {
      console.error('Unexpected error:', error)
      resolve(NextResponse.json({ error: 'Unexpected error occurred' }, { status: 500 }))
    }
  })
} 