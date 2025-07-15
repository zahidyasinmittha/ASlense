import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Home from './pages/Home';
import Learn from './pages/Learn';
import Practice from './pages/Practice';
import Translate from './pages/Translate';
import About from './pages/About';
import Contact from './pages/Contact';
import CSVImporter from './pages/CsvImporter';

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/learn" element={<Learn />} />
          <Route path="/practice" element={<Practice />} />
          <Route path="/translate" element={<Translate />} />
          <Route path="/about" element={<About />} />
          <Route path="/contact" element={<Contact />} />
          <Route path="/csv" element={<CSVImporter />} />

        </Routes>
      </Layout>
    </Router>
  );
}

export default App;