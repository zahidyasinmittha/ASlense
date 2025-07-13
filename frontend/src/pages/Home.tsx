import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, Brain, Video, Bot, Hand, Users, Star, CheckCircle, Play, Sparkles, Target, Award, Rocket, Globe, Heart } from 'lucide-react';

const Home: React.FC = () => {
  const [currentTestimonial, setCurrentTestimonial] = useState(0);
  const [isVisible, setIsVisible] = useState(false);
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

  useEffect(() => {
    setIsVisible(true);
    const interval = setInterval(() => {
      setCurrentTestimonial((prev) => (prev + 1) % testimonials.length);
    }, 5000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      setMousePosition({ x: e.clientX, y: e.clientY });
    };
    window.addEventListener('mousemove', handleMouseMove);
    return () => window.removeEventListener('mousemove', handleMouseMove);
  }, []);

  const features = [
    {
      icon: Brain,
      title: 'Learn Signs',
      description: 'Master ASL with our comprehensive video lessons. From basic alphabet to complex phrases, learn at your own pace with interactive content.',
      color: 'blue',
      link: '/learn',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      icon: Video,
      title: 'Practice Gestures',
      description: 'Perfect your signing with our AI-powered gesture recognition. Get real-time feedback and track your progress as you improve.',
      color: 'purple',
      link: '/practice',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: Bot,
      title: 'Translate in Real-Time',
      description: 'Bridge communication gaps with our live translation feature. Convert sign language to text instantly using advanced AI technology.',
      color: 'green',
      link: '/translate',
      gradient: 'from-green-500 to-emerald-500'
    }
  ];

  const testimonials = [
    {
      name: 'Sarah Johnson',
      role: 'ASL Student',
      content: 'ASLense has revolutionized how I learn sign language. The real-time feedback is incredible!',
      rating: 5,
      avatar: '👩‍🎓',
      color: 'blue'
    },
    {
      name: 'Michael Chen',
      role: 'Deaf Community Member',
      content: 'Finally, a tool that truly understands the nuances of ASL. This is a game-changer.',
      rating: 5,
      avatar: '👨‍💼',
      color: 'purple'
    },
    {
      name: 'Emma Rodriguez',
      role: 'Educator',
      content: 'I use ASLense in my classroom. My students love the interactive learning experience.',
      rating: 5,
      avatar: '👩‍🏫',
      color: 'green'
    }
  ];

  const stats = [
    { number: '500+', label: 'Signs Available', color: 'blue', icon: '🤟' },
    { number: '10k+', label: 'Active Learners', color: 'purple', icon: '👥' },
    { number: '95%', label: 'Recognition Accuracy', color: 'green', icon: '🎯' },
    { number: '24/7', label: 'Availability', color: 'orange', icon: '⏰' }
  ];

  const achievements = [
    { icon: Award, title: 'AI Innovation Award 2024', color: 'yellow' },
    { icon: Target, title: 'Accessibility Excellence', color: 'blue' },
    { icon: Users, title: 'Community Choice', color: 'purple' },
    { icon: Sparkles, title: 'Best Learning Platform', color: 'pink' }
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
    <div className="min-h-screen overflow-hidden">
      {/* Hero Section */}
      <section className="py-4 relative min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 overflow-hidden">
        {/* Interactive Background */}
        <div className="absolute inset-0">
          {/* Animated mesh gradient */}
          <div className="absolute inset-0 opacity-30">
            <div className="absolute top-0 -left-4 w-72 h-72 bg-purple-300 rounded-full mix-blend-multiply filter blur-xl animate-blob"></div>
            <div className="absolute top-0 -right-4 w-72 h-72 bg-yellow-300 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-2000"></div>
            <div className="absolute -bottom-8 left-20 w-72 h-72 bg-pink-300 rounded-full mix-blend-multiply filter blur-xl animate-blob animation-delay-4000"></div>
          </div>

          {/* Floating ASL hand signs */}
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

          {/* Interactive cursor trail */}
          <div
            className="absolute w-6 h-6 bg-blue-400 rounded-full opacity-20 pointer-events-none transition-all duration-300 ease-out"
            style={{
              left: mousePosition.x - 12,
              top: mousePosition.y - 12,
              transform: 'scale(0.5)',
            }}
          />

          {/* Geometric patterns */}
          <div className="absolute inset-0 opacity-10">
            {[...Array(20)].map((_, i) => (
              <div
                key={i}
                className="absolute w-2 h-2 bg-white rounded-full animate-pulse"
                style={{
                  left: `${Math.random() * 100}%`,
                  top: `${Math.random() * 100}%`,
                  animationDelay: `${Math.random() * 3}s`,
                }}
              />
            ))}
          </div>
        </div>

        <div className="relative max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8 flex items-center min-h-screen">
          <div className="w-full">
            {/* Main Hero Content */}
            <div className="text-center lg:text-left lg:flex lg:items-center lg:justify-between">
              <div className="lg:w-1/2">
                <div className={`transition-all duration-1000 delay-300 ${isVisible ? 'animate-fade-in-up' : 'opacity-0 translate-y-10'}`}>
                  <div className="inline-flex items-center px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-white/80 text-sm font-medium mb-6 animate-shimmer">
                    <Rocket className="h-4 w-4 mr-2 text-yellow-400" />
                    <span>Powered by Advanced AI Technology</span>
                  </div>
                </div>
                
                <h1 className={`text-5xl md:text-7xl lg:text-8xl font-bold mb-6 transition-all duration-1000 delay-500 ${isVisible ? 'animate-fade-in-up' : 'opacity-0 translate-y-10'}`}>
                  <span className="text-white block">Bridge</span>
                  <span className="text-transparent bg-clip-text bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 animate-gradient-x">
                    Communication
                  </span>
                  <span className="text-white/90 block text-4xl md:text-5xl lg:text-6xl mt-2">
                    with ASL
                  </span>
                </h1>
                
                <p className={`text-xl md:text-2xl text-white/80 mb-8 max-w-2xl leading-relaxed transition-all duration-1000 delay-700 ${isVisible ? 'animate-fade-in-up' : 'opacity-0 translate-y-10'}`}>
                  Experience the future of sign language learning with AI-powered gesture recognition, 
                  real-time translation, and interactive lessons designed for everyone.
                </p>
                
                <div className={`flex flex-col sm:flex-row gap-6 justify-center lg:justify-start items-center transition-all duration-1000 delay-900 ${isVisible ? 'animate-fade-in-up' : 'opacity-0 translate-y-10'}`}>
                  <Link
                    to="/learn"
                    className="group relative inline-flex items-center px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 text-white font-semibold rounded-2xl hover:from-blue-700 hover:to-purple-700 transition-all duration-300 shadow-2xl hover:shadow-blue-500/25 transform hover:scale-105 overflow-hidden"
                  >
                    <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 transform -skew-x-12 -translate-x-full group-hover:translate-x-full transition-transform duration-1000"></div>
                    <Play className="mr-3 h-6 w-6 group-hover:scale-110 transition-transform duration-300 relative z-10" />
                    <span className="relative z-10">Start Learning</span>
                    <ArrowRight className="ml-3 h-6 w-6 group-hover:translate-x-2 transition-transform duration-300 relative z-10" />
                  </Link>
                  
                  <Link
                    to="/translate"
                    className="group inline-flex items-center px-8 py-4 bg-white/10 backdrop-blur-sm text-white font-semibold rounded-2xl border border-white/20 hover:bg-white/20 transition-all duration-300 transform hover:scale-105"
                  >
                    <Globe className="mr-3 h-6 w-6 group-hover:rotate-12 transition-transform duration-300" />
                    <span>Try Translation</span>
                  </Link>
                </div>

                {/* Achievement Badges */}
                <div className={`flex flex-wrap justify-center lg:justify-start gap-4 mt-12 transition-all duration-1000 delay-1100 ${isVisible ? 'animate-fade-in-up' : 'opacity-0 translate-y-10'}`}>
                  {achievements.map((achievement, index) => {
                    const Icon = achievement.icon;
                    return (
                      <div
                        key={index}
                        className={`flex items-center space-x-2 px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full border border-white/20 text-white text-sm font-medium hover:scale-105 transition-transform duration-300 animate-scale-in`}
                        style={{ animationDelay: `${1200 + index * 100}ms` }}
                      >
                        <Icon className={`h-4 w-4 text-${achievement.color}-400`} />
                        <span>{achievement.title}</span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Interactive 3D Hand Animation */}
              <div className="lg:w-1/2 mt-12 lg:mt-0 flex justify-center lg:justify-end">
                <div className={`relative transition-all duration-1000 delay-1000 ${isVisible ? 'animate-scale-in' : 'opacity-0 scale-75'}`}>
                  <div className="relative group">
                    {/* Glowing background */}
                    <div className="absolute -inset-8 bg-gradient-to-r from-blue-600 via-purple-600 to-pink-600 rounded-full blur-2xl opacity-30 group-hover:opacity-50 transition-opacity duration-1000 animate-pulse-glow"></div>
                    
                    {/* Main hand container */}
                    <div className="relative w-80 h-80 flex items-center justify-center">
                      {/* Rotating rings */}
                      <div className="absolute inset-0 border-2 border-blue-400/30 rounded-full animate-spin" style={{ animationDuration: '20s' }}></div>
                      <div className="absolute inset-4 border-2 border-purple-400/30 rounded-full animate-spin" style={{ animationDuration: '15s', animationDirection: 'reverse' }}></div>
                      <div className="absolute inset-8 border-2 border-pink-400/30 rounded-full animate-spin" style={{ animationDuration: '10s' }}></div>
                      
                      {/* Central hand icon */}
                      <div className="relative z-10 text-8xl animate-bounce-gentle group-hover:scale-110 transition-transform duration-500">
                        <Hand className="h-32 w-32 text-white drop-shadow-2xl animate-wiggle" />
                      </div>
                      
                      {/* Floating particles around hand */}
                      {[...Array(8)].map((_, i) => (
                        <div
                          key={i}
                          className="absolute w-3 h-3 bg-gradient-to-r from-blue-400 to-purple-400 rounded-full animate-float"
                          style={{
                            left: `${50 + 40 * Math.cos((i * Math.PI * 2) / 8)}%`,
                            top: `${50 + 40 * Math.sin((i * Math.PI * 2) / 8)}%`,
                            animationDelay: `${i * 0.5}s`,
                            animationDuration: '3s'
                          }}
                        />
                      ))}
                    </div>
                    
                    {/* Interactive elements */}
                    <div className="absolute -bottom-4 left-1/2 transform -translate-x-1/2 text-center">
                      <div className="px-4 py-2 bg-white/10 backdrop-blur-sm rounded-full text-white text-sm font-medium animate-pulse">
                        AI-Powered Recognition
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 transform -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 border-2 border-white/30 rounded-full flex justify-center">
            <div className="w-1 h-3 bg-white/50 rounded-full mt-2 animate-pulse"></div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-white relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 animate-fade-in-up">
              Everything You Need to Master ASL
            </h2>
            <p className="text-xl text-gray-600 animate-fade-in-up animation-delay-200">
              Comprehensive tools for learning, practicing, and translating sign language
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 stagger-animation">
            {features.map((feature, index) => {
              const Icon = feature.icon;
              return (
                <div
                  key={index}
                  className={`group relative bg-white p-8 rounded-3xl shadow-lg hover:shadow-2xl transition-all duration-500 transform hover:scale-105 card-hover overflow-hidden`}
                >
                  {/* Animated Background */}
                  <div className={`absolute inset-0 bg-gradient-to-br ${feature.gradient} opacity-0 group-hover:opacity-10 transition-opacity duration-500`}></div>
                  
                  {/* Floating Icon */}
                  <div className={`relative flex items-center justify-center w-20 h-20 bg-gradient-to-br ${feature.gradient} rounded-2xl mb-6 group-hover:scale-110 group-hover:rotate-3 transition-all duration-500 shadow-lg`}>
                    <Icon className="h-10 w-10 text-white animate-float" style={{ animationDelay: `${index * 500}ms` }} />
                    <div className="absolute -inset-2 bg-gradient-to-r from-white/20 to-transparent rounded-2xl opacity-0 group-hover:opacity-100 transition-opacity duration-300"></div>
                  </div>
                  
                  <h3 className="text-2xl font-bold text-gray-900 mb-4 group-hover:text-blue-600 transition-colors duration-300">
                    {feature.title}
                  </h3>
                  
                  <p className="text-gray-600 mb-6 leading-relaxed">
                    {feature.description}
                  </p>
                  
                  <Link
                    to={feature.link}
                    className={`inline-flex items-center text-${feature.color}-600 font-medium hover:text-${feature.color}-700 transition-all duration-300 group-hover:translate-x-2`}
                  >
                    Get Started
                    <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform duration-300" />
                  </Link>

                  {/* Decorative Elements */}
                  <div className="absolute top-4 right-4 w-20 h-20 bg-gradient-to-br from-blue-100 to-purple-100 rounded-full opacity-20 group-hover:scale-150 transition-transform duration-700"></div>
                  <div className="absolute bottom-4 left-4 w-12 h-12 bg-gradient-to-br from-pink-100 to-yellow-100 rounded-full opacity-30 group-hover:scale-125 transition-transform duration-500"></div>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Statistics Section */}
      <section className="py-20 bg-gradient-to-br from-gray-900 via-blue-900 to-purple-900 relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          {[...Array(50)].map((_, i) => (
            <div
              key={i}
              className="absolute w-1 h-1 bg-white/20 rounded-full animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 5}s`,
                animationDuration: `${3 + Math.random() * 4}s`
              }}
            ></div>
          ))}
        </div>

        <div className="relative max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-white mb-4 animate-fade-in-up">
              Empowering the Deaf Community
            </h2>
            <p className="text-xl text-blue-200 animate-fade-in-up animation-delay-200">
              Join thousands of learners bridging the communication gap
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8 stagger-animation">
            {stats.map((stat, index) => (
              <div
                key={index}
                className={`text-center p-8 bg-white/10 backdrop-blur-sm rounded-2xl border border-white/20 hover:bg-white/20 transition-all duration-500 transform hover:scale-105 glass-dark group`}
              >
                <div className="text-4xl mb-4 group-hover:animate-bounce">{stat.icon}</div>
                <div className={`text-4xl md:text-5xl font-bold text-${stat.color}-400 mb-2 animate-counter group-hover:animate-pulse`}>
                  {stat.number}
                </div>
                <div className="text-gray-300 font-medium">{stat.label}</div>
                <div className={`w-full h-1 bg-gradient-to-r from-${stat.color}-500 to-${stat.color}-300 rounded-full mt-4 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-500`}></div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Testimonials Section */}
      <section className="py-20 bg-white relative overflow-hidden">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4 animate-fade-in-up">
              What Our Users Say
            </h2>
            <p className="text-xl text-gray-600 animate-fade-in-up animation-delay-200">
              Real feedback from our amazing community
            </p>
          </div>

          {/* Testimonial Carousel */}
          <div className="relative max-w-4xl mx-auto">
            <div className="overflow-hidden rounded-3xl">
              <div 
                className="flex transition-transform duration-500 ease-in-out"
                style={{ transform: `translateX(-${currentTestimonial * 100}%)` }}
              >
                {testimonials.map((testimonial, index) => (
                  <div key={index} className="w-full flex-shrink-0 px-8">
                    <div className={`bg-gradient-to-br from-${testimonial.color}-50 to-${testimonial.color}-100 p-8 md:p-12 rounded-2xl shadow-lg text-center`}>
                      <div className="flex items-center justify-center mb-6">
                        {[...Array(testimonial.rating)].map((_, i) => (
                          <Star key={i} className="h-6 w-6 text-yellow-400 fill-current animate-pulse" style={{ animationDelay: `${i * 100}ms` }} />
                        ))}
                      </div>
                      
                      <blockquote className="text-xl md:text-2xl text-gray-700 mb-8 italic leading-relaxed">
                        "{testimonial.content}"
                      </blockquote>
                      
                      <div className="flex items-center justify-center space-x-4">
                        <div className={`w-16 h-16 bg-gradient-to-br from-${testimonial.color}-400 to-${testimonial.color}-600 rounded-full flex items-center justify-center text-2xl animate-bounce-gentle`}>
                          {testimonial.avatar}
                        </div>
                        <div className="text-left">
                          <div className="font-semibold text-gray-900 text-lg">{testimonial.name}</div>
                          <div className="text-gray-600">{testimonial.role}</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Testimonial Indicators */}
            <div className="flex justify-center space-x-2 mt-8">
              {testimonials.map((_, index) => (
                <button
                  key={index}
                  onClick={() => setCurrentTestimonial(index)}
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    index === currentTestimonial 
                      ? 'bg-blue-600 scale-125' 
                      : 'bg-gray-300 hover:bg-gray-400'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 bg-gradient-to-br from-blue-600 via-purple-600 to-pink-600 relative overflow-hidden">
        {/* Animated Background */}
        <div className="absolute inset-0">
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-r from-blue-600/20 to-purple-600/20 animate-gradient-x"></div>
          {[...Array(30)].map((_, i) => (
            <div
              key={i}
              className="absolute w-2 h-2 bg-white/10 rounded-full animate-float"
              style={{
                left: `${Math.random() * 100}%`,
                top: `${Math.random() * 100}%`,
                animationDelay: `${Math.random() * 3}s`,
                animationDuration: `${2 + Math.random() * 3}s`
              }}
            ></div>
          ))}
        </div>

        <div className="relative max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <div className="animate-fade-in-up">
            <div className="flex justify-center mb-8">
              <div className="relative">
                <Heart className="h-16 w-16 text-red-400 animate-heartbeat" />
                <div className="absolute inset-0 animate-ping">
                  <Heart className="h-16 w-16 text-red-400 opacity-30" />
                </div>
              </div>
            </div>
            
            <h2 className="text-3xl md:text-5xl font-bold text-white mb-6 animate-fade-in-up animation-delay-200">
              Ready to Start Your ASL Journey?
            </h2>
            <p className="text-xl md:text-2xl text-blue-100 mb-8 animate-fade-in-up animation-delay-400">
              Join thousands of learners and start communicating in American Sign Language today.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-6 justify-center animate-fade-in-up animation-delay-600">
              <Link
                to="/learn"
                className="group inline-flex items-center px-10 py-5 bg-white text-blue-600 font-medium rounded-2xl hover:bg-gray-50 transition-all duration-300 shadow-2xl hover:shadow-3xl transform hover:scale-105 btn-bounce"
              >
                <CheckCircle className="mr-3 h-6 w-6 group-hover:animate-pulse" />
                Start Learning Now
              </Link>
              
              <Link
                to="/about"
                className="group inline-flex items-center px-10 py-5 bg-transparent text-white font-medium rounded-2xl border-2 border-white hover:bg-white hover:text-blue-600 transition-all duration-300 transform hover:scale-105"
              >
                Learn More About Us
                <ArrowRight className="ml-3 h-6 w-6 group-hover:translate-x-1 transition-transform duration-300" />
              </Link>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

export default Home;