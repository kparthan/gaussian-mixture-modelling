#ifndef STRUCTURE_H 
#define STRUCTURE_H 

#include "Header.h"

class Structure
{
  private:
    //! Structure identifier
    string name;

    std::vector<Vector> coordinates;

  protected:
    //! Reads the profile from a file
    void read_profile(string &);

  public: 
    //! Null constructor
    Structure();

    //! Constructor
    Structure(std::vector<Vector> &, string &);

    //! Loads the spherical system
    void load(string &);

    //! Loads the spherical system
    void load(path &);

    //! Gets the list of unit coordinates
    std::vector<Vector> getCoordinates();
};

#endif

