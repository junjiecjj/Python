/**
 * Count the number of elements in the array
 * @param {Array} array Array of numbers
 * @returns {number} The number of elements in the array
 */
 const countLoop = (array) => {
    let result = 0;

    if (array.length === 0) {
      return result;
    }

    for (let i = 0; i < array.length; i += 1) {
      result += 1;
    }

    return result;
 }

  console.log(countLoop([3,5,8,11])) //4
  console.log(countLoop([])) // 0