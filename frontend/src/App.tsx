import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import ChatInterface from './components/ChatInterface';
import Sidebar from './components/Sidebar';
import './App.css';

function App() {
  return (
    <Router>
      <div style={{ display: 'flex', height: '100vh', overflow: 'hidden' }}>
        <Sidebar />
        <Routes>
          <Route path="/" element={<ChatInterface />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
