import './App.css'
import Sidebar from './components/Sidebar'
import Chat from './components/Chat'
import Header from './components/Header'

function App() {
  return (
    <div id="root">
      <Header />
      <div className="content-wrapper">
        <Sidebar />
        <Chat />
      </div>
    </div>
  )
}

export default App
