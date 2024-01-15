/**
 * Calculate the largest number
 * This solution works for arrays of any length
 * @param {Array} array Array of numbers
 * @returns {number} The argest number
 */
 const maxLoop = (array) => {
    let result = 0;

    if (array.length === 0) {
      return result;
    }

    for (let i = 0; i < array.length; i += 1) {
      if (array[i] > result) {
        result = array[i]
      }
    }
    return result;
  };

  console.log(maxLoop([2,5,7,22,11])); // 22
  console.log(maxLoop([])) // 0