import React from "react";
import { motion } from "framer-motion";

const FeaturesSection = ({
  title = "Feature Title",
  description = "Plan your meals, worem ipsum dolor sit amet",
  bulletPoints = [],
  buttonText = "Learn More",
  buttonAction = () => {},
  imageSrc = "https://via.placeholder.com/428x460",
  reversed = false,
}) => {
  // Animation variants



const containerVariants = {
  hidden: { opacity: 0, y: 50 },
  visible: {
    opacity: 1,
    y: 0,
    transition: {
      duration: 0.8,
      staggerChildren: 0.2
    }
  },
  exit: { opacity: 0, y: -50 }
};

const textVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6 }
  }
};

  return (
   <motion.div
      className="w-full px-6 md:px-16 lg:px-20 py-16 md:py-24 bg-gradient-to-br from-gray-50 to-white flex flex-col items-center relative overflow-hidden"
      initial="hidden"
      whileInView="visible"
      exit="exit"
      viewport={{ once: false, amount: 0.2 }}
      variants={containerVariants}
    >
      {/* Background accent */}
      <div className="absolute top-0 right-0 w-96 h-96 bg-gradient-to-bl from-yellow-50 to-transparent rounded-full blur-3xl opacity-30" />
      
      <div
        className={`w-full max-w-7xl flex flex-col lg:flex-row ${
          reversed ? "lg:flex-row-reverse" : ""
        } items-center gap-12 lg:gap-20 relative z-10`}
      >
        {/* Text Section */}
        <motion.div
          className="flex-1 flex flex-col gap-8 text-center lg:text-left"
          variants={textVariants}
        >
          {/* Title with accent */}
          <div className="relative">
            <motion.h3 
              className="text-sm md:text-base font-semibold text-amber-600 tracking-wide uppercase mb-2"
              initial={{ opacity: 0, x: -20 }}
              whileInView={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              {title}
            </motion.h3>
            <div className="w-12 h-0.5 bg-gradient-to-r from-amber-400 to-yellow-500 lg:mx-0 mx-auto" />
          </div>

          {/* Main heading */}
          <motion.h2 
            className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-light text-gray-900 leading-tight"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            {description}
          </motion.h2>

          {/* Bullet points with enhanced styling */}
          <motion.div
            className="space-y-4"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            {bulletPoints.map((point, index) => (
              <motion.div
                key={index}
                className="flex items-start gap-3 text-left"
                initial={{ opacity: 0, x: -20 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
              >
                <div className="w-1.5 h-1.5 bg-gradient-to-r from-amber-400 to-yellow-500 rounded-full mt-2.5 flex-shrink-0" />
                <p className="text-base md:text-lg text-gray-700 leading-relaxed font-light">
                  {point}
                </p>
              </motion.div>
            ))}
          </motion.div>

          {/* Professional button */}
          <motion.div
            className="flex lg:justify-start justify-center"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.6 }}
          >
            <button
              onClick={buttonAction}
              className="group relative px-8 py-4 bg-gradient-to-r from-amber-400 to-yellow-500 rounded-full text-gray-900 text-base md:text-lg font-medium shadow-lg hover:shadow-xl transition-all duration-300 transform hover:scale-105 hover:from-amber-500 hover:to-yellow-600 overflow-hidden"
            >
              <span className="relative z-10 flex items-center gap-2">
                {buttonText}
                <svg 
                  className="w-5 h-5 transition-transform group-hover:translate-x-1" 
                  fill="none" 
                  stroke="currentColor" 
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 8l4 4m0 0l-4 4m4-4H3" />
                </svg>
              </span>
              <div className="absolute inset-0 bg-white opacity-0 group-hover:opacity-10 transition-opacity duration-300" />
            </button>
          </motion.div>
        </motion.div>

        {/* Image Section */}
        <motion.div
          className="flex-1 flex justify-center relative"
          variants={textVariants}
        >
          <div className="relative group">
            {/* Subtle background glow */}
            <div className="absolute -inset-4 bg-gradient-to-r from-amber-200 to-yellow-200 rounded-3xl blur-2xl opacity-20 group-hover:opacity-30 transition-opacity duration-500" />
            
            {/* Main image container */}
            <div className="relative bg-white rounded-2xl p-4 shadow-2xl group-hover:shadow-3xl transition-all duration-500 transform group-hover:scale-105">
              <img
                className="w-full max-w-md h-auto rounded-xl"
                src={imageSrc}
                alt="Professional Feature Illustration"
              />
            </div>

            {/* Floating accent elements */}
            <div className="absolute -top-2 -right-2 w-6 h-6 bg-gradient-to-br from-amber-400 to-yellow-500 rounded-full opacity-80" />
            <div className="absolute -bottom-4 -left-4 w-4 h-4 bg-gradient-to-br from-yellow-400 to-amber-500 rounded-full opacity-60" />
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
};

export default FeaturesSection;
