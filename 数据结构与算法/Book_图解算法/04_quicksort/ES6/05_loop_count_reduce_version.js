  /**
 * Count the number of elements in the array
 * @param {Array} array Array of numbers
 * @returns {number} The number of elements in the array
 */
   const countReduce = (array) => array.reduce((count, next) => count += 1, 0);

   console.log(countReduce([5,8,11,13])) // 4
   console.log(countReduce([]))// 0