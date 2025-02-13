import { useState } from 'react'
import Evaluation from '../pages/Evaluation'
import './App.css'

function App() {
    const [data, setData] = useState(null)

    return (
      <>
        <div>
        <Evaluation />
        </div>
      </>
    )
}

export default App
