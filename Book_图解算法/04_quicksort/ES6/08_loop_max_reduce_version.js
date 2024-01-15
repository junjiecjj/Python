/**
 * Calculate the largest number
 * This solution works for arrays of any length
 * @param {Array} array Array of numbers
 * @returns {number} The argest number
 */
 const maxReduce = (array) => array.reduce((curr, next) => next > curr ? next : curr, 0);

 console.log(maxReduce([1,3,11,7])) // 11
 console.log(maxReduce([])) // 0