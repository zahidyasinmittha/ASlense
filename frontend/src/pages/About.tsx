import React from 'react';
import { Target, Users, Zap, Heart, Award, Lightbulb, Shield } from 'lucide-react';

const About: React.FC = () => {
  const technologies = [
    { name: 'React', icon: '⚛️', description: 'Modern UI framework', color: 'blue' },
    { name: 'TypeScript', icon: '🔷', description: 'Type-safe development', color: 'blue' },
    { name: 'Tailwind CSS', icon: '🎨', description: 'Utility-first CSS', color: 'cyan' },
    { name: 'MediaPipe', icon: '🤖', description: 'ML gesture recognition', color: 'green' },
    { name: 'OpenPose', icon: '🎯', description: 'Human pose estimation', color: 'purple' },
    { name: 'WebRTC', icon: '📹', description: 'Real-time communication', color: 'red' },
  ];

  const teamMembers = [
    { name: 'Sarah Johnson', role: 'Lead Developer', avatar: '👩‍💻', color: 'blue' },
    { name: 'Michael Chen', role: 'ML Engineer', avatar: '👨‍🔬', color: 'green' },
    { name: 'Emma Rodriguez', role: 'UX Designer', avatar: '👩‍🎨', color: 'purple' },
    { name: 'David Kim', role: 'ASL Consultant', avatar: '👨‍🏫', color: 'orange' },
  ];

  const values = [
    {
      icon: Shield,
      title: 'Accessibility',
      description: 'We believe technology should be inclusive and accessible to everyone, regardless of their abilities.',
      color: 'blue'
    },
    {
      icon: Lightbulb,
      title: 'Innovation',
      description: 'We continuously push the boundaries of what\'s possible with AI and machine learning.',
      color: 'yellow'
    },
    {
      icon: Users,
      title: 'Community',
      description: 'We work closely with the deaf community to ensure our solutions meet real needs.',
      color: 'green'
    }
  ];

  const stats = [
    { number: '500+', label: 'ASL Signs Supported', icon: '🤟', color: 'blue' },
    { number: '10,000+', label: 'Active Users', icon: '👥', color: 'purple' },
    { number: '95%', label: 'Recognition Accuracy', icon: '🎯', color: 'green' },
    { number: '24/7', label: 'Availability', icon: '⏰', color: 'red' },
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
      {/* Mission Section */}
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
              <Target className="relative h-16 w-16 text-blue-600 mx-auto animate-pulse" />
            </div>
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-6">Our Mission</h1>
            <p className="text-xl text-gray-600 leading-relaxed">
              ASLense is dedicated to breaking down communication barriers between the deaf and hearing communities. 
              We leverage cutting-edge AI technology to make American Sign Language more accessible, 
              creating a world where everyone can communicate effectively regardless of their hearing abilities.
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

      {/* How It Works */}
      <section className="py-20 bg-white">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">How It Works</h2>
            <p className="text-xl text-gray-600">
              Advanced AI technology meets intuitive design
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {[
              { emoji: '📹', title: 'Capture', description: 'Our system uses your device\'s camera to capture hand movements and gestures in real-time.', color: 'blue' },
              { emoji: '🧠', title: 'Analyze', description: 'Advanced machine learning models analyze the gestures and identify the corresponding signs.', color: 'purple' },
              { emoji: '💬', title: 'Translate', description: 'The system converts the recognized signs into text, enabling seamless communication.', color: 'green' }
            ].map((step, index) => (
              <div
                key={index}
                className={`group text-center p-8 rounded-2xl bg-gradient-to-br from-${step.color}-50 to-${step.color}-100 hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 200}ms` }}
              >
                <div className={`bg-${step.color}-100 rounded-full w-20 h-20 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <span className="text-3xl">{step.emoji}</span>
                </div>
                <h3 className={`text-xl font-semibold text-${step.color}-900 mb-4 group-hover:text-${step.color}-700 transition-colors duration-300`}>
                  {step.title}
                </h3>
                <p className={`text-${step.color}-700 leading-relaxed`}>
                  {step.description}
                </p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Technologies */}
      <section className="py-20 bg-gray-50">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 bg-gradient-to-r from-yellow-400 to-orange-500 rounded-full blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Zap className="relative h-12 w-12 text-yellow-500 mx-auto animate-bounce" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Powered by Modern Technology</h2>
            <p className="text-xl text-gray-600">
              Built with cutting-edge tools and frameworks
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {technologies.map((tech, index) => (
              <div
                key={index}
                className={`group bg-white rounded-xl shadow-sm p-6 hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 100}ms` }}
              >
                <div className="flex items-center space-x-4">
                  <div className={`text-4xl p-3 bg-${tech.color}-50 rounded-xl group-hover:scale-110 transition-transform duration-300`}>
                    {tech.icon}
                  </div>
                  <div>
                    <h3 className={`text-lg font-semibold text-gray-900 group-hover:text-${tech.color}-600 transition-colors duration-300`}>
                      {tech.name}
                    </h3>
                    <p className="text-gray-600">{tech.description}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Team */}
      <section className="py-20 bg-white">
        <div className="max-w-[90vw]  mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Users className="relative h-12 w-12 text-blue-600 mx-auto animate-pulse" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Meet Our Team</h2>
            <p className="text-xl text-gray-600">
              Passionate individuals working to bridge communication gaps
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {teamMembers.map((member, index) => (
              <div
                key={index}
                className={`group text-center p-6 bg-gradient-to-br from-${member.color}-50 to-${member.color}-100 rounded-2xl hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 150}ms` }}
              >
                <div className={`bg-gradient-to-br from-${member.color}-400 to-${member.color}-600 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-4 group-hover:scale-110 transition-transform duration-300 shadow-lg`}>
                  <span className="text-4xl">{member.avatar}</span>
                </div>
                <h3 className={`text-lg font-semibold text-${member.color}-900 mb-2 group-hover:text-${member.color}-700 transition-colors duration-300`}>
                  {member.name}
                </h3>
                <p className={`text-${member.color}-700`}>{member.role}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Values */}
      <section className="py-20 bg-gradient-to-br from-purple-50 to-pink-50">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 bg-gradient-to-r from-red-500 to-pink-500 rounded-full blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Heart className="relative h-16 w-16 text-red-500 mx-auto animate-pulse" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Our Values</h2>
            <p className="text-xl text-gray-600">
              The principles that guide everything we do
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {values.map((value, index) => {
              const Icon = value.icon;
              return (
                <div
                  key={index}
                  className={`group text-center p-8 bg-white rounded-2xl shadow-sm hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                  style={{ animationDelay: `${index * 200}ms` }}
                >
                  <div className={`bg-${value.color}-100 rounded-full w-16 h-16 flex items-center justify-center mx-auto mb-6 group-hover:scale-110 transition-transform duration-300`}>
                    <Icon className={`h-8 w-8 text-${value.color}-600`} />
                  </div>
                  <h3 className={`text-xl font-semibold text-gray-900 mb-4 group-hover:text-${value.color}-600 transition-colors duration-300`}>
                    {value.title}
                  </h3>
                  <p className="text-gray-600 leading-relaxed">
                    {value.description}
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Statistics */}
      <section className="py-20 bg-white">
        <div className="max-w-[90vw] mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16 animate-fade-in-up">
            <div className="relative group mb-6">
              <div className="absolute -inset-2 bg-gradient-to-r from-green-500 to-blue-500 rounded-full blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <Award className="relative h-12 w-12 text-green-600 mx-auto animate-bounce" />
            </div>
            <h2 className="text-3xl md:text-4xl font-bold text-gray-900 mb-4">Making a Difference</h2>
            <p className="text-xl text-gray-600">
              Numbers that showcase our impact
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div
                key={index}
                className={`group text-center p-8 bg-gradient-to-br from-${stat.color}-50 to-${stat.color}-100 rounded-2xl shadow-sm hover:shadow-xl transition-all duration-500 transform hover:scale-105 animate-fade-in-up`}
                style={{ animationDelay: `${index * 150}ms` }}
              >
                <div className="text-4xl mb-4 group-hover:scale-110 transition-transform duration-300">
                  {stat.icon}
                </div>
                <div className={`text-4xl font-bold text-${stat.color}-600 mb-2 animate-counter`}>
                  {stat.number}
                </div>
                <div className={`text-${stat.color}-700 font-medium`}>{stat.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>
    </div>
  );
};

export default About;