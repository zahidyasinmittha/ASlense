import React, { useState } from 'react';
import { Mail, Phone, MapPin, Send, Github, Linkedin, Twitter, MessageSquare, Clock, CheckCircle, AlertCircle } from 'lucide-react';
import { contactAPI } from '../services/api';

const Contact: React.FC = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState<'idle' | 'success' | 'error'>('idle');

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      // Send contact form data to backend using API service
      const response = await contactAPI.submitContact(formData);

      if (response.status === 200 || response.status === 201) {
        setSubmitStatus('success');
        
        // Reset form after success
        setTimeout(() => {
          setFormData({ name: '', email: '', subject: '', message: '' });
          setSubmitStatus('idle');
        }, 3000);
      } else {
        setSubmitStatus('error');
      }
    } catch (error) {
      console.error('Error submitting contact form:', error);
      setSubmitStatus('error');
    } finally {
      setIsSubmitting(false);
    }
  };

  const contactInfo = [
    {
      icon: Mail,
      title: 'Email',
      details: ['hello@aslense.com', 'support@aslense.com'],
      color: 'blue'
    },
    {
      icon: Phone,
      title: 'Phone',
      details: ['+1 (555) 123-4567', 'Mon-Fri 9AM-6PM EST'],
      color: 'green'
    },
    {
      icon: MapPin,
      title: 'Office',
      details: ['123 Innovation Drive', 'San Francisco, CA 94105'],
      color: 'purple'
    }
  ];

  const socialLinks = [
    { icon: Github, href: '#', label: 'GitHub', color: 'text-gray-700 hover:text-gray-900 bg-gray-100 hover:bg-gray-200' },
    { icon: Linkedin, href: '#', label: 'LinkedIn', color: 'text-blue-600 hover:text-blue-700 bg-blue-100 hover:bg-blue-200' },
    { icon: Twitter, href: '#', label: 'Twitter', color: 'text-blue-400 hover:text-blue-500 bg-blue-100 hover:bg-blue-200' },
  ];

  const faqs = [
    {
      question: 'How accurate is the gesture recognition?',
      answer: 'Our AI models achieve 95% accuracy in controlled conditions, with continuous improvements being made.',
      color: 'blue'
    },
    {
      question: 'Is ASLense free to use?',
      answer: 'Yes! ASLense is completely free and open to everyone who wants to learn or use ASL.',
      color: 'green'
    },
    {
      question: 'Do you support other sign languages?',
      answer: 'Currently we focus on ASL, but we\'re planning to add support for other sign languages in the future.',
      color: 'purple'
    }
  ];

  const floatingElements = [
    { emoji: '🤟', delay: 0, duration: 4 },
    { emoji: '👋', delay: 1, duration: 5 },
    { emoji: '✋', delay: 2, duration: 3 },
    { emoji: '👌', delay: 3, duration: 6 },
    { emoji: '🤲', delay: 4, duration: 4 },
    { emoji: '👐', delay: 5, duration: 5 },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <section className="relative bg-gradient-to-br from-blue-50 via-purple-50 to-pink-50 py-20 overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-20 left-20 w-32 h-32 bg-blue-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob"></div>
          <div className="absolute bottom-20 right-20 w-32 h-32 bg-purple-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-2000"></div>
          <div className="absolute top-40 right-40 w-32 h-32 bg-pink-200 rounded-full mix-blend-multiply filter blur-xl opacity-70 animate-blob animation-delay-4000"></div>
        </div>

        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="animate-fade-in-up">
            <div className="relative group mb-8">
              <div className="absolute -inset-4 blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <MessageSquare className="relative h-16 w-16 text-blue-600 mx-auto animate-pulse" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">Get In Touch</h1>
            <p className="text-xl text-gray-600">
              Have questions about ASLense? We'd love to hear from you. 
              Send us a message and we'll respond as soon as possible.
            </p>
             {floatingElements.map((element, i) => (
            <div
              key={i}
              className="absolute text-4xl opacity-20 animate-float pointer-events-none"
              style={{
                left: `${10 + (i * 15)}%`,
                top: `${20 + (i * 10)}%`,
                animationDelay: `${element.delay}s`,
                animationDuration: `${element.duration}s`
              }}
            >
              {element.emoji}
            </div>
          ))}
           {/* Rotating rings */}
                      <div className="absolute inset-0 border-2 border-blue-400/30 rounded-full animate-spin" style={{ animationDuration: '20s' }}></div>
                      <div className="absolute inset-4 border-2 border-purple-400/30 rounded-full animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }}></div>
                      <div className="absolute inset-8 border-2 border-pink-400/30 rounded-full animate-spin" style={{ animationDuration: '10s' }}></div>
          </div>
        </div>
      </section>

      <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
          {/* Contact Information */}
          <div className="lg:col-span-1 space-y-8">
            <div className="bg-white rounded-xl shadow-sm p-8 animate-fade-in-up animation-delay-200">
              <h2 className="text-2xl font-bold text-gray-900 mb-8">Contact Information</h2>
              
              <div className="space-y-6">
                {contactInfo.map((info, index) => {
                  const Icon = info.icon;
                  return (
                    <div key={index} className="group flex items-start space-x-4">
                      <div className={`bg-${info.color}-100 rounded-lg p-3 group-hover:scale-110 transition-transform duration-300`}>
                        <Icon className={`h-6 w-6 text-${info.color}-600`} />
                      </div>
                      <div>
                        <h3 className={`font-semibold text-gray-900 group-hover:text-${info.color}-600 transition-colors duration-300`}>
                          {info.title}
                        </h3>
                        {info.details.map((detail, i) => (
                          <p key={i} className="text-gray-600">{detail}</p>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Social Links */}
              <div className="mt-8 pt-8 border-t border-gray-200">
                <h3 className="font-semibold text-gray-900 mb-4">Follow Us</h3>
                <div className="flex space-x-4">
                  {socialLinks.map((link, index) => {
                    const Icon = link.icon;
                    return (
                      <a
                        key={index}
                        href={link.href}
                        className={`p-3 rounded-lg transition-all duration-300 transform hover:scale-110 ${link.color}`}
                        aria-label={link.label}
                      >
                        <Icon className="h-6 w-6" />
                      </a>
                    );
                  })}
                </div>
              </div>

              {/* Office Hours */}
              <div className="mt-8 p-4 bg-gradient-to-br from-blue-50 to-purple-50 rounded-lg">
                <div className="flex items-center space-x-2 mb-2">
                  <Clock className="h-5 w-5 text-blue-600" />
                  <h3 className="font-semibold text-blue-900">Office Hours</h3>
                </div>
                <div className="text-sm text-blue-800 space-y-1">
                  <div>Monday - Friday: 9:00 AM - 6:00 PM PST</div>
                  <div>Saturday: 10:00 AM - 4:00 PM PST</div>
                  <div>Sunday: Closed</div>
                </div>
              </div>
            </div>
          </div>

          {/* Contact Form */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-xl shadow-sm p-8 animate-fade-in-up animation-delay-400">
              <h2 className="text-2xl font-bold text-gray-900 mb-8">Send Us a Message</h2>
              
              <form onSubmit={handleSubmit} className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-2">
                      Full Name *
                    </label>
                    <input
                      type="text"
                      id="name"
                      name="name"
                      value={formData.name}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:border-gray-400"
                      placeholder="Enter your full name"
                      required
                    />
                  </div>
                  
                  <div>
                    <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-2">
                      Email Address *
                    </label>
                    <input
                      type="email"
                      id="email"
                      name="email"
                      value={formData.email}
                      onChange={handleInputChange}
                      className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:border-gray-400"
                      placeholder="Enter your email address"
                      required
                    />
                  </div>
                </div>

                <div>
                  <label htmlFor="subject" className="block text-sm font-medium text-gray-700 mb-2">
                    Subject *
                  </label>
                  <select
                    id="subject"
                    name="subject"
                    value={formData.subject}
                    onChange={handleInputChange}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:border-gray-400"
                    required
                  >
                    <option value="">Select a subject</option>
                    <option value="general">General Inquiry</option>
                    <option value="support">Technical Support</option>
                    <option value="feedback">Feedback</option>
                    <option value="partnership">Partnership</option>
                    <option value="other">Other</option>
                  </select>
                </div>

                <div>
                  <label htmlFor="message" className="block text-sm font-medium text-gray-700 mb-2">
                    Message *
                  </label>
                  <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleInputChange}
                    rows={6}
                    className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all duration-300 hover:border-gray-400 resize-none"
                    placeholder="Tell us about your project, questions, or how we can help you..."
                    required
                  />
                </div>

                <div className="flex items-center justify-between">
                  <div className="text-sm text-gray-500">
                    {submitStatus === 'success' ? (
                      <div className="flex items-center text-green-600">
                        <CheckCircle className="h-4 w-4 mr-2" />
                        Message sent successfully!
                      </div>
                    ) : submitStatus === 'error' ? (
                      <div className="flex items-center text-red-600">
                        <AlertCircle className="h-4 w-4 mr-2" />
                        Failed to send message. Please try again.
                      </div>
                    ) : (
                      "We'll get back to you within 24 hours."
                    )}
                  </div>
                  <button
                    type="submit"
                    disabled={isSubmitting || submitStatus === 'success'}
                    className="group inline-flex items-center px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium rounded-lg hover:from-blue-700 hover:to-purple-700 focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 transition-all duration-300 transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg hover:shadow-xl"
                  >
                    {isSubmitting ? (
                      <>
                        <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                        Sending...
                      </>
                    ) : submitStatus === 'success' ? (
                      <>
                        <CheckCircle className="h-5 w-5 mr-2" />
                        Sent!
                      </>
                    ) : (
                      <>
                        <Send className="h-5 w-5 mr-2 group-hover:translate-x-1 transition-transform duration-300" />
                        Send Message
                      </>
                    )}
                  </button>
                </div>
              </form>
            </div>

            {/* FAQ Section */}
            <div className="mt-8 bg-gradient-to-br from-blue-50 to-purple-50 rounded-xl p-8 animate-fade-in-up animation-delay-600">
              <h3 className="text-xl font-bold text-gray-900 mb-6">Frequently Asked Questions</h3>
              <div className="space-y-6">
                {faqs.map((faq, index) => (
                  <div key={index} className="group">
                    <h4 className={`font-semibold text-gray-900 mb-2 group-hover:text-${faq.color}-600 transition-colors duration-300`}>
                      {faq.question}
                    </h4>
                    <p className="text-gray-700 leading-relaxed">{faq.answer}</p>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Contact;