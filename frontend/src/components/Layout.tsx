import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Menu, X, Hand, Mail, Phone, MapPin, Github, Linkedin, Twitter, Heart, User, LogOut, Settings } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const location = useLocation();
  const { user, logout, isAdmin } = useAuth();

  const navItems = [
    { name: 'Home', path: '/' },
    { name: 'Learn', path: '/learn' },
    { name: 'Practice', path: '/practice' },
    // { name: 'Translate', path: '/translate' }, // Commented out per user request
    { name: 'About', path: '/about' },
    { name: 'Contact', path: '/contact' },
  ];

  const toggleMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  const footerLinks = {
    product: [
      { name: 'Learn ASL', path: '/learn' },
      { name: 'Practice Signs', path: '/practice' },
      { name: 'Live Translation', path: '/translate' },
    ],
    company: [
      { name: 'About Us', path: '/about' },
      { name: 'Contact', path: '/contact' },
      { name: 'Privacy Policy', path: '#' },
      { name: 'Terms of Service', path: '#' },
    ],
    resources: [
      { name: 'Documentation', path: '#' },
      { name: 'API Reference', path: '#' },
      { name: 'Community', path: '#' },
      { name: 'Support', path: '#' },
    ],
  };

  const socialLinks = [
    { icon: Github, href: '#', label: 'GitHub' },
    { icon: Linkedin, href: '#', label: 'LinkedIn' },
    { icon: Twitter, href: '#', label: 'Twitter' },
  ];

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* Navigation */}
      <nav className="bg-white/95 backdrop-blur-md shadow-sm sticky top-0 z-50 border-b border-gray-100">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            {/* Logo */}
            <Link to="/" className="flex items-center space-x-2 group">
              <div className="relative">
                <Hand className="h-8 w-8 text-blue-600 group-hover:text-blue-700 transition-all duration-300 group-hover:scale-110" />
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-purple-500 rounded-full opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
              </div>
              <span className="text-xl font-bold text-gray-900 group-hover:text-blue-600 transition-colors duration-300">ASLense</span>
            </Link>

            {/* Desktop Navigation */}
            <div className="hidden md:flex items-center space-x-1">
              {navItems.map((item, index) => (
                <Link
                  key={item.name}
                  to={item.path}
                  className={`relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-300 hover:scale-105 ${
                    location.pathname === item.path
                      ? 'text-blue-600 bg-blue-50 shadow-sm'
                      : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50'
                  }`}
                  style={{ animationDelay: `${index * 100}ms` }}
                >
                  {item.name}
                  {location.pathname === item.path && (
                    <div className="absolute bottom-0 left-1/2 transform -translate-x-1/2 w-1 h-1 bg-blue-600 rounded-full"></div>
                  )}
                </Link>
              ))}
              
              {/* Authentication buttons */}
              <div className="ml-4 flex items-center space-x-3">
                {user ? (
                  <div className="flex items-center space-x-3">
                    <Link
                      to="/dashboard"
                      className="flex items-center space-x-2 px-3 py-2 text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-gray-50 rounded-lg transition-all duration-300"
                    >
                      <User className="h-4 w-4" />
                      <span>{user.username}</span>
                    </Link>
                    {isAdmin && (
                      <Link
                        to="/admin"
                        className="px-3 py-2 text-sm font-medium text-red-700 hover:text-red-800 hover:bg-red-50 rounded-lg transition-all duration-300"
                      >
                        <Settings className="h-4 w-4 inline mr-1" />
                        Admin
                      </Link>
                    )}
                    <button
                      onClick={logout}
                      className="flex items-center space-x-1 px-3 py-2 text-sm font-medium text-gray-700 hover:text-red-600 hover:bg-gray-50 rounded-lg transition-all duration-300"
                    >
                      <LogOut className="h-4 w-4" />
                      <span>Logout</span>
                    </button>
                  </div>
                ) : (
                  <>
                    <Link
                      to="/login"
                      className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-blue-600 hover:bg-gray-50 rounded-lg transition-all duration-300"
                    >
                      Sign In
                    </Link>
                    <Link
                      to="/register"
                      className="px-4 py-2 text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 rounded-lg transition-all duration-300 shadow-sm"
                    >
                      Sign Up
                    </Link>
                  </>
                )}
              </div>
            </div>

            {/* Mobile menu button */}
            <div className="md:hidden">
              <button
                onClick={toggleMenu}
                className="inline-flex items-center justify-center p-2 rounded-md text-gray-700 hover:text-blue-600 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500 transition-all duration-300"
              >
                <div className="relative w-6 h-6">
                  <Menu className={`absolute inset-0 h-6 w-6 transition-all duration-300 ${isMenuOpen ? 'opacity-0 rotate-180' : 'opacity-100 rotate-0'}`} />
                  <X className={`absolute inset-0 h-6 w-6 transition-all duration-300 ${isMenuOpen ? 'opacity-100 rotate-0' : 'opacity-0 -rotate-180'}`} />
                </div>
              </button>
            </div>
          </div>
        </div>

        {/* Mobile menu */}
        <div className={`md:hidden transition-all duration-300 ease-in-out ${isMenuOpen ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'} overflow-hidden`}>
          <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3 bg-white/95 backdrop-blur-md border-t border-gray-100">
            {navItems.map((item, index) => (
              <Link
                key={item.name}
                to={item.path}
                onClick={() => setIsMenuOpen(false)}
                className={`block px-3 py-2 rounded-md text-base font-medium transition-all duration-300 transform hover:scale-105 ${
                  location.pathname === item.path
                    ? 'text-blue-600 bg-blue-50'
                    : 'text-gray-700 hover:text-blue-600 hover:bg-gray-50'
                }`}
                style={{ 
                  animationDelay: `${index * 50}ms`,
                  transform: isMenuOpen ? 'translateX(0)' : 'translateX(-20px)',
                  transition: `all 300ms ease-out ${index * 50}ms`
                }}
              >
                {item.name}
              </Link>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {children}
      </main>

      {/* Footer */}
      <footer className="bg-gray-900 text-white">
        <div className="max-w-[90vw]  mx-auto px-4 sm:px-6 lg:px-8 py-12">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-8">
            {/* Brand Section */}
            <div className="lg:col-span-2">
              <div className="flex items-center space-x-2 mb-4">
                <Hand className="h-8 w-8 text-blue-400" />
                <span className="text-xl font-bold">ASLense</span>
              </div>
              <p className="text-gray-300 mb-6 max-w-md">
                Bridging communication gaps through AI-powered sign language recognition. 
                Making ASL accessible to everyone, everywhere.
              </p>
              <div className="flex space-x-4">
                {socialLinks.map((social, index) => {
                  const Icon = social.icon;
                  return (
                    <a
                      key={index}
                      href={social.href}
                      className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-all duration-300 hover:scale-110 group"
                      aria-label={social.label}
                    >
                      <Icon className="h-5 w-5 text-gray-400 group-hover:text-white transition-colors duration-300" />
                    </a>
                  );
                })}
              </div>
            </div>

            {/* Product Links */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-blue-400">Product</h3>
              <ul className="space-y-2">
                {footerLinks.product.map((link, index) => (
                  <li key={index}>
                    <Link
                      to={link.path}
                      className="text-gray-300 hover:text-white transition-colors duration-300 hover:translate-x-1 transform inline-block"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Company Links */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-purple-400">Company</h3>
              <ul className="space-y-2">
                {footerLinks.company.map((link, index) => (
                  <li key={index}>
                    <Link
                      to={link.path}
                      className="text-gray-300 hover:text-white transition-colors duration-300 hover:translate-x-1 transform inline-block"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>

            {/* Resources Links */}
            <div>
              <h3 className="text-lg font-semibold mb-4 text-green-400">Resources</h3>
              <ul className="space-y-2">
                {footerLinks.resources.map((link, index) => (
                  <li key={index}>
                    <Link
                      to={link.path}
                      className="text-gray-300 hover:text-white transition-colors duration-300 hover:translate-x-1 transform inline-block"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          {/* Contact Info */}
          <div className="border-t border-gray-800 mt-8 pt-8">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="flex items-center space-x-3 group">
                <div className="p-2 bg-blue-600 rounded-lg group-hover:bg-blue-500 transition-colors duration-300">
                  <Mail className="h-4 w-4 text-white" />
                </div>
                <div>
                  <div className="text-sm text-gray-400">Email</div>
                  <div className="text-white">hello@aslense.com</div>
                </div>
              </div>
              <div className="flex items-center space-x-3 group">
                <div className="p-2 bg-green-600 rounded-lg group-hover:bg-green-500 transition-colors duration-300">
                  <Phone className="h-4 w-4 text-white" />
                </div>
                <div>
                  <div className="text-sm text-gray-400">Phone</div>
                  <div className="text-white">+1 (555) 123-4567</div>
                </div>
              </div>
              <div className="flex items-center space-x-3 group">
                <div className="p-2 bg-purple-600 rounded-lg group-hover:bg-purple-500 transition-colors duration-300">
                  <MapPin className="h-4 w-4 text-white" />
                </div>
                <div>
                  <div className="text-sm text-gray-400">Location</div>
                  <div className="text-white">San Francisco, CA</div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Bar */}
          <div className="border-t border-gray-800 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
            <div className="text-gray-400 text-sm mb-4 md:mb-0">
              © 2025 ASLense. All rights reserved.
            </div>
            <div className="flex items-center space-x-2 text-gray-400 text-sm">
              <span>Made with</span>
              <Heart className="h-4 w-4 text-red-500 animate-pulse" />
              <span>for the deaf community</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Layout;